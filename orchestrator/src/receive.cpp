#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#define COMMAND_DURATION_MS 3500

#define POSITION_TOLERANCE 0.01  // meters
#define ANGULAR_TOLERANCE 0.05   // radians
#define LINEAR_SPEED 0.3
#define ANGULAR_SPEED 0.5
#define ROTATION_LINEAR_SPEED 0.1  // Slow forward speed while rotating

// Obstacle avoidance parameters
#define OBSTACLE_DETECTION_DISTANCE 0.4  // meters
#define ROTATION_OBSTACLE_DETECTION_DISTANCE (OBSTACLE_DETECTION_DISTANCE * 0.5f)
#define OBSTACLE_CLEAR_DISTANCE 0.4      // meters - distance considered "clear"
#define OBSTACLE_MARGIN_ANGLE 0.1        // radians - margin from obstacle when finding clear path
#define AVOIDANCE_FORWARD_TIME 3.0       // seconds to go forward after obstacle avoidance

using std::placeholders::_1;

enum NavigationState {
    IDLE,
    ROTATING_TO_GOAL,
    MOVING_TO_GOAL,
    OBSTACLE_DETECTED,
    OBSTACLE_ROTATING,
    OBSTACLE_MOVING_FORWARD
};

class MinimalSubscriber : public rclcpp::Node {
public:
    MinimalSubscriber() : Node("minimal_subscriber"), 
                          is_navigating_(false),
                          has_goal_(false),
                          initial_x_(0.0), 
                          initial_y_(0.0),
                          initial_yaw_(0.0),
                          current_x_(0.0),
                          current_y_(0.0),
                          current_yaw_(0.0),
                          nav_state_(IDLE),
                          avoidance_start_time_(0.0) {
        
        cmd_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/baiter_cmd", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
        
        goal_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/baiter_goal_pos", 10, std::bind(&MinimalSubscriber::goal_callback, this, _1));
        
        odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&MinimalSubscriber::odom_callback, this, _1));
        
        scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&MinimalSubscriber::scan_callback, this, _1));
        
        cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10);
        
        stop_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(COMMAND_DURATION_MS),
            std::bind(&MinimalSubscriber::stop_robot, this));
        stop_timer_->cancel();
        
        navigation_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&MinimalSubscriber::navigation_control_loop, this));
        navigation_timer_->cancel();
    }

private:
    // ========== Callback Functions ==========
    
    void topic_callback(const std_msgs::msg::String& msg) {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg.data.c_str());
        
        auto twist_msg = geometry_msgs::msg::Twist();
        
        if (msg.data == "avance") {
            twist_msg.linear.x = 0.5;
            twist_msg.angular.z = 0.0;
        }
        else if (msg.data == "recule") {
            twist_msg.linear.x = 0.0;
            twist_msg.angular.z = 0.0;
            stop_navigation();
        }
        else if (msg.data == "gauche") {
            twist_msg.linear.x = 0.0;
            twist_msg.angular.z = 0.5;
        }
        else if (msg.data == "droite") {
            twist_msg.linear.x = 0.0;
            twist_msg.angular.z = -0.5;
        }
        else if (msg.data == "lucien") {
            handle_lucien_command();
            return;
        }
        else {
            RCLCPP_WARN(this->get_logger(), "Unknown command: '%s'", msg.data.c_str());
            return;
        }
        
        cmd_vel_publisher_->publish(twist_msg);
        RCLCPP_INFO(this->get_logger(), "Published cmd_vel");
        
        stop_timer_->reset();
    }
    
    void goal_callback(const geometry_msgs::msg::Point& msg) {
        if (!is_navigating_) {
            goal_x_ = msg.x;
            goal_y_ = msg.y;
            has_goal_ = true;
            RCLCPP_INFO(this->get_logger(), "New goal received: (%.2f, %.2f)", goal_x_, goal_y_);
        } else {
            RCLCPP_WARN(this->get_logger(), "Robot is navigating, ignoring new goal");
        }
    }
    
    void odom_callback(const nav_msgs::msg::Odometry& msg) {
        current_x_ = msg.pose.pose.position.x;
        current_y_ = msg.pose.pose.position.y;
        
        current_yaw_ = extract_yaw_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        );
        
        if (!is_navigating_) {
            initial_x_ = current_x_;
            initial_y_ = current_y_;
            initial_yaw_ = current_yaw_;
        }
    }
    
    void scan_callback(const sensor_msgs::msg::LaserScan& msg) {
        scan_data_ = msg;
        has_scan_data_ = true;
    }
    
    // ========== Navigation Functions ==========
    
    void handle_lucien_command() {
        if (!has_goal_) {
            RCLCPP_WARN(this->get_logger(), "No goal position available for 'lucien' command");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Starting autonomous navigation to goal");
        is_navigating_ = true;
        nav_state_ = ROTATING_TO_GOAL;
        stop_timer_->cancel();
        navigation_timer_->reset();
    }
    
    void navigation_control_loop() {
        if (!is_navigating_) {
            return;
        }
        
        // Transform goal to global frame
        double cos_yaw = std::cos(initial_yaw_);
        double sin_yaw = std::sin(initial_yaw_);
        
        double rotated_goal_x = goal_x_ * cos_yaw - goal_y_ * sin_yaw;
        double rotated_goal_y = goal_x_ * sin_yaw + goal_y_ * cos_yaw;
        
        double absolute_goal_x = initial_x_ + rotated_goal_x;
        double absolute_goal_y = initial_y_ + rotated_goal_y;
        
        double distance = compute_distance_to_goal(absolute_goal_x, absolute_goal_y);
        double angle_to_goal = compute_angle_to_goal(absolute_goal_x, absolute_goal_y);
        
        // Check if goal is reached
        if (distance < POSITION_TOLERANCE) {
            stop_navigation();
            RCLCPP_INFO(this->get_logger(), "Goal reached!");
            return;
        }
        
        // State machine for navigation
        auto twist_msg = geometry_msgs::msg::Twist();
        
        switch (nav_state_) {
            case ROTATING_TO_GOAL:
                handle_rotating_to_goal(angle_to_goal, twist_msg);
                break;
                
            case MOVING_TO_GOAL:
                handle_moving_to_goal(distance, angle_to_goal, twist_msg);
                break;
                
            case OBSTACLE_DETECTED:
                handle_obstacle_detected(twist_msg);
                break;
                
            case OBSTACLE_ROTATING:
                handle_obstacle_rotating(twist_msg);
                break;
                
            case OBSTACLE_MOVING_FORWARD:
                handle_obstacle_moving_forward(twist_msg);
                break;
                
            default:
                break;
        }
        
        cmd_vel_publisher_->publish(twist_msg);
    }
    
    void handle_rotating_to_goal(double angle_to_goal, geometry_msgs::msg::Twist& twist_msg) {
        // Check for obstacles
        if (has_scan_data_ && detect_obstacle(ROTATION_OBSTACLE_DETECTION_DISTANCE)) {
            RCLCPP_INFO(this->get_logger(), "Obstacle detected during rotation!");
            nav_state_ = OBSTACLE_DETECTED;
            return;
        }
        
        // Rotate towards goal while moving slowly forward
        if (std::abs(angle_to_goal) > ANGULAR_TOLERANCE) {
            twist_msg.linear.x = ROTATION_LINEAR_SPEED;  // Slow forward motion
            twist_msg.angular.z = (angle_to_goal > 0) ? ANGULAR_SPEED : -ANGULAR_SPEED;
            RCLCPP_INFO(this->get_logger(), "Rotating to goal (angle: %.2f)", angle_to_goal);
        } else {
            // Aligned with goal, switch to moving state
            nav_state_ = MOVING_TO_GOAL;
            RCLCPP_INFO(this->get_logger(), "Aligned with goal, moving forward");
        }
    }
    
    void handle_moving_to_goal(double distance, double angle_to_goal, geometry_msgs::msg::Twist& twist_msg) {
        // Check for obstacles
        if (has_scan_data_ && detect_obstacle(OBSTACLE_DETECTION_DISTANCE)) {
            RCLCPP_INFO(this->get_logger(), "Obstacle detected while moving!");
            nav_state_ = OBSTACLE_DETECTED;
            return;
        }
        
        // If we've drifted off course, go back to rotating
        if (std::abs(angle_to_goal) > ANGULAR_TOLERANCE * 2.0) {
            nav_state_ = ROTATING_TO_GOAL;
            RCLCPP_INFO(this->get_logger(), "Off course, rotating again");
            return;
        }
        
        // Move forward at full speed with minor corrections
        twist_msg.linear.x = std::min(LINEAR_SPEED, distance);
        twist_msg.angular.z = angle_to_goal * 2.0;  // Proportional correction
        RCLCPP_INFO(this->get_logger(), "Moving to goal (dist: %.2f)", distance);
    }
    
    void handle_obstacle_detected(geometry_msgs::msg::Twist& twist_msg) {
        // Stop and analyze which side is clearer
        twist_msg.linear.x = 0.0;
        twist_msg.angular.z = 0.0;
        
        bool turn_left = is_left_side_clearer();
        avoidance_turn_left_ = turn_left;
        
        RCLCPP_INFO(this->get_logger(), "Obstacle avoidance: turning %s", 
                    turn_left ? "LEFT" : "RIGHT");
        
        nav_state_ = OBSTACLE_ROTATING;
    }
    
    void handle_obstacle_rotating(geometry_msgs::msg::Twist& twist_msg) {
        // Rotate until path is clear
        if (is_path_clear_with_margin()) {
            RCLCPP_INFO(this->get_logger(), "Path clear, moving forward for avoidance");
            nav_state_ = OBSTACLE_MOVING_FORWARD;
            avoidance_start_time_ = this->now().seconds();
            return;
        }
        
        // Continue rotating
        twist_msg.linear.x = 0.0;
        twist_msg.angular.z = avoidance_turn_left_ ? ANGULAR_SPEED : -ANGULAR_SPEED;
    }
    
    void handle_obstacle_moving_forward(geometry_msgs::msg::Twist& twist_msg) {
        double elapsed_time = this->now().seconds() - avoidance_start_time_;
        
        if (elapsed_time >= AVOIDANCE_FORWARD_TIME) {
            RCLCPP_INFO(this->get_logger(), "Avoidance complete, resuming goal navigation");
            nav_state_ = ROTATING_TO_GOAL;
            return;
        }
        
        // Check if we hit another obstacle while avoiding
        if (has_scan_data_ && detect_obstacle(OBSTACLE_DETECTION_DISTANCE)) {
            RCLCPP_INFO(this->get_logger(), "Another obstacle during avoidance!");
            nav_state_ = OBSTACLE_DETECTED;
            return;
        }
        
        // Move forward
        twist_msg.linear.x = LINEAR_SPEED;
        twist_msg.angular.z = 0.0;
    }
    
    void stop_navigation() {
        is_navigating_ = false;
        nav_state_ = IDLE;
        navigation_timer_->cancel();
        
        auto twist_msg = geometry_msgs::msg::Twist();
        twist_msg.linear.x = 0.0;
        twist_msg.angular.z = 0.0;
        cmd_vel_publisher_->publish(twist_msg);
    }
    
    void stop_robot() {
        auto twist_msg = geometry_msgs::msg::Twist();
        twist_msg.linear.x = 0.0;
        twist_msg.angular.z = 0.0;
        
        cmd_vel_publisher_->publish(twist_msg);
        RCLCPP_INFO(this->get_logger(), "Auto-stop: Published zero cmd_vel");
        
        stop_timer_->cancel();
    }
    
    // ========== Obstacle Detection Functions ==========
    
    bool detect_obstacle(float detection_distance) {
        if (!has_scan_data_ || scan_data_.ranges.empty()) {
            return false;
        }
        
        // Check front cone (±30 degrees)
        int num_readings = scan_data_.ranges.size();
        int cone_size = num_readings / 8;  // ~30 degrees on each side
        
        int valid_readings = 0;
        int obstacle_readings = 0;
        
        for (int i = 0; i < cone_size; i++) {
            float range = scan_data_.ranges[i];
            if (std::isfinite(range) && range > 0.1) {  // Ignore noise near 0
                valid_readings++;
                if (range < detection_distance) {
                    obstacle_readings++;
                }
            }
        }
        
        for (int i = num_readings - cone_size; i < num_readings; i++) {
            float range = scan_data_.ranges[i];
            if (std::isfinite(range) && range > 0.1) {  // Ignore noise near 0
                valid_readings++;
                if (range < detection_distance) {
                    obstacle_readings++;
                }
            }
        }
        
        // Detect obstacle if at least 30% of valid readings show an obstacle
        if (valid_readings == 0) {
            return false;
        }
        
        float obstacle_ratio = static_cast<float>(obstacle_readings) / valid_readings;
        return obstacle_ratio >= 0.3;
    }
    
    bool is_left_side_clearer() {
        if (!has_scan_data_ || scan_data_.ranges.empty()) {
            return true;  // default
        }

        float left_min = std::numeric_limits<float>::infinity();
        float right_min = std::numeric_limits<float>::infinity();

        for (size_t i = 0; i < scan_data_.ranges.size(); ++i) {
            float r = scan_data_.ranges[i];
            if (!std::isfinite(r)) continue;
            if (r < scan_data_.range_min || r > scan_data_.range_max) continue;

            float angle = scan_data_.angle_min + i * scan_data_.angle_increment;

            // RIGHT: -90° to -30°
            if (angle > -M_PI / 2 && angle < -M_PI / 6) {
                right_min = std::min(right_min, r);
            }

            // LEFT: +30° to +90°
            else if (angle > M_PI / 6 && angle < M_PI / 2) {
                left_min = std::min(left_min, r);
            }
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Left min: %.2f, Right min: %.2f",
            left_min,
            right_min
        );

        // Prefer the side with more clearance
        return left_min > right_min;
    }
        
    bool is_path_clear_with_margin() {
        if (!has_scan_data_ || scan_data_.ranges.empty()) {
            return false;
        }
        
        // Check front cone (±30 degrees) with larger clearance distance
        int num_readings = scan_data_.ranges.size();
        int cone_size = num_readings / 8;  // ~30 degrees on each side
        
        int valid_readings = 0;
        int clear_readings = 0;
        
        for (int i = 0; i < cone_size; i++) {
            float range = scan_data_.ranges[i];
            if (std::isfinite(range) && range > 0.1) {  // Ignore noise near 0
                valid_readings++;
                if (range >= OBSTACLE_CLEAR_DISTANCE) {
                    clear_readings++;
                }
            }
        }
        
        for (int i = num_readings - cone_size; i < num_readings; i++) {
            float range = scan_data_.ranges[i];
            if (std::isfinite(range) && range > 0.1) {  // Ignore noise near 0
                valid_readings++;
                if (range >= OBSTACLE_CLEAR_DISTANCE) {
                    clear_readings++;
                }
            }
        }
        
        // Path is clear if at least 80% of valid readings show clearance
        if (valid_readings == 0) {
            return false;
        }
        
        float clear_ratio = static_cast<float>(clear_readings) / valid_readings;
        RCLCPP_INFO(this->get_logger(), "Clear ratio: %.2f (%d/%d)", 
                    clear_ratio, clear_readings, valid_readings);
        
        return clear_ratio >= 0.8;
    }
    
    // ========== Utility Functions ==========
    
    double compute_distance_to_goal(double goal_x, double goal_y) {
        double dx = goal_x - current_x_;
        double dy = goal_y - current_y_;
        return std::sqrt(dx * dx + dy * dy);
    }
    
    double compute_angle_to_goal(double goal_x, double goal_y) {
        double dx = goal_x - current_x_;
        double dy = goal_y - current_y_;
        double desired_yaw = std::atan2(dy, dx);
        double angle_diff = normalize_angle(desired_yaw - current_yaw_);
        return angle_diff;
    }
    
    double extract_yaw_from_quaternion(double x, double y, double z, double w) {
        double siny_cosp = 2.0 * (w * z + x * y);
        double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        return std::atan2(siny_cosp, cosy_cosp);
    }
    
    double normalize_angle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
    
    // ========== Member Variables ==========
    
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cmd_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr goal_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::TimerBase::SharedPtr stop_timer_;
    rclcpp::TimerBase::SharedPtr navigation_timer_;
    
    bool is_navigating_;
    bool has_goal_;
    bool has_scan_data_ = false;
    
    NavigationState nav_state_;
    bool avoidance_turn_left_;
    double avoidance_start_time_;
    
    double goal_x_, goal_y_;
    double initial_x_, initial_y_, initial_yaw_;
    double current_x_, current_y_, current_yaw_;
    
    sensor_msgs::msg::LaserScan scan_data_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}
