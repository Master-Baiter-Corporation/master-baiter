#include <memory>
#include <cmath>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "nav_msgs/msg/odometry.hpp"

#define COMMAND_DURATION_MS 3500

#define POSITION_TOLERANCE 0.01  // meters
#define ANGULAR_TOLERANCE 0.05   // radians
#define LINEAR_SPEED 0.3
#define ANGULAR_SPEED 0.5

using std::placeholders::_1;

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
                          current_yaw_(0.0) {
        
        cmd_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/baiter_cmd", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
        
        goal_subscription_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/baiter_goal_pos", 10, std::bind(&MinimalSubscriber::goal_callback, this, _1));
        
        odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&MinimalSubscriber::odom_callback, this, _1));
        
        cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10);
        
        stop_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(COMMAND_DURATION_MS),
            std::bind(&MinimalSubscriber::stop_robot, this));
        stop_timer_->cancel();
        
        // Timer for navigation control loop
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
            twist_msg.linear.x = -0.5;
            twist_msg.angular.z = 0.0;
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
            return;  // Don't use the stop timer for navigation
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
        // Only accept new goals when not navigating
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
        
        // Keep updating initial position and orientation until navigation starts
        if (!is_navigating_) {
            initial_x_ = current_x_;
            initial_y_ = current_y_;
            initial_yaw_ = current_yaw_;
        }
    }
    
    // ========== Navigation Functions ==========
    
    void handle_lucien_command() {
        if (!has_goal_) {
            RCLCPP_WARN(this->get_logger(), "No goal position available for 'lucien' command");
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Starting autonomous navigation to goal");
        is_navigating_ = true;
        stop_timer_->cancel();  // Cancel manual stop timer
        navigation_timer_->reset();  // Start navigation control loop
    }
    
    void navigation_control_loop() {
        if (!is_navigating_) {
            return;
        }
        
        // Transform goal from robot's local frame to global frame
        // Goal (goal_x_, goal_y_) is given in the robot's frame at command time
        // We need to rotate it by initial_yaw_ and translate it by initial position
        double cos_yaw = std::cos(initial_yaw_);
        double sin_yaw = std::sin(initial_yaw_);
        
        double rotated_goal_x = goal_x_ * cos_yaw - goal_y_ * sin_yaw;
        double rotated_goal_y = goal_x_ * sin_yaw + goal_y_ * cos_yaw;
        
        double absolute_goal_x = initial_x_ + rotated_goal_x;
        double absolute_goal_y = initial_y_ + rotated_goal_y;
        
        // Compute distance and angle to goal
        double distance = compute_distance_to_goal(absolute_goal_x, absolute_goal_y);
        double angle_to_goal = compute_angle_to_goal(absolute_goal_x, absolute_goal_y);
        
        RCLCPP_INFO(this->get_logger(), "Distance: %.2f, Angle: %.2f", distance, angle_to_goal);
        
        // Check if goal is reached
        if (distance < POSITION_TOLERANCE) {
            stop_navigation();
            RCLCPP_INFO(this->get_logger(), "Goal reached!");
            return;
        }
        
        // Compute and publish velocity command
        auto twist_msg = compute_velocity_command(distance, angle_to_goal);
        cmd_vel_publisher_->publish(twist_msg);
    }
    
    void stop_navigation() {
        is_navigating_ = false;
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
        
        // Compute angle difference (normalized to [-pi, pi])
        double angle_diff = normalize_angle(desired_yaw - current_yaw_);
        return angle_diff;
    }
    
    geometry_msgs::msg::Twist compute_velocity_command(double distance, double angle_to_goal) {
        auto twist_msg = geometry_msgs::msg::Twist();
        
        // If angle error is large, rotate in place first
        if (std::abs(angle_to_goal) > ANGULAR_TOLERANCE) {
            twist_msg.linear.x = 0.0;
            twist_msg.angular.z = (angle_to_goal > 0) ? ANGULAR_SPEED : -ANGULAR_SPEED;
        } else {
            // Move forward while making small corrections
            twist_msg.linear.x = std::min(LINEAR_SPEED, distance);  // Slow down near goal
            twist_msg.angular.z = angle_to_goal * 2.0;  // Proportional correction
        }
        
        return twist_msg;
    }
    
    double extract_yaw_from_quaternion(double x, double y, double z, double w) {
        // Convert quaternion to yaw angle
        double siny_cosp = 2.0 * (w * z + x * y);
        double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        return std::atan2(siny_cosp, cosy_cosp);
    }
    
    double normalize_angle(double angle) {
        // Normalize angle to [-pi, pi]
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }
    
    // ========== Member Variables ==========
    
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cmd_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr goal_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::TimerBase::SharedPtr stop_timer_;
    rclcpp::TimerBase::SharedPtr navigation_timer_;
    
    bool is_navigating_;
    bool has_goal_;
    
    double goal_x_, goal_y_;
    double initial_x_, initial_y_, initial_yaw_;
    double current_x_, current_y_, current_yaw_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}
