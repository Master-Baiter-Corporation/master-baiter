#include <memory>
#include <cmath>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "odas_ros_msgs/msg/odas_ssl.hpp"

#define MIN_DISTANCE 0.5   // meters - minimum estimated distance
#define MAX_DISTANCE 3.0   // meters - maximum estimated distance
#define MIN_ENERGY 0.1     // minimum energy threshold to consider valid

using std::placeholders::_1;

class SoundGoalPublisher : public rclcpp::Node {
public:
    SoundGoalPublisher() : Node("sound_goal_publisher") {
        
        // Subscribe to sound source localization
        ssl_subscription_ = this->create_subscription<odas_ros_msgs::msg::OdasSsl>(
            "/ssl", 10, std::bind(&SoundGoalPublisher::ssl_callback, this, _1));
        
        // Publisher for goal position
        goal_publisher_ = this->create_publisher<geometry_msgs::msg::Point>(
            "/baiter_goal_pos", 10);
        
        RCLCPP_INFO(this->get_logger(), "Sound Goal Publisher node started");
    }

private:
    // ========== Callback Functions ==========
    
    void ssl_callback(const odas_ros_msgs::msg::OdasSsl& msg) {
        // Extract 3D unit vector components
        double x = msg.x;
        double y = msg.y;
        double z = msg.z;
        double energy = msg.e;
        
        RCLCPP_INFO(this->get_logger(), 
                    "Received SSL: x=%.3f, y=%.3f, z=%.3f, e=%.3f", 
                    x, y, z, energy);
        
        // Filter out low energy sounds
        if (energy < MIN_ENERGY) {
            RCLCPP_WARN(this->get_logger(), 
                        "Energy too low (%.3f < %.3f), ignoring sound", 
                        energy, MIN_ENERGY);
            return;
        }
        
        // Project 3D vector onto 2D plane (ignore z component)
        double x_2d = x;
        double y_2d = y;
        
        // Normalize the 2D projection to get unit vector in 2D
        double norm_2d = std::sqrt(x_2d * x_2d + y_2d * y_2d);
        
        if (norm_2d < 0.01) {  // Nearly vertical sound source
            RCLCPP_WARN(this->get_logger(), 
                        "Sound source is nearly vertical, ignoring");
            return;
        }
        
        x_2d /= norm_2d;
        y_2d /= norm_2d;
        
        // Estimate distance based on energy
        // Higher energy = closer sound source
        // Energy ranges from 0 to 1, map to distance range
        double estimated_distance = estimate_distance_from_energy(energy);
        
        // Compute goal position (distance along the 2D direction vector)
        double goal_x = x_2d * estimated_distance;
        double goal_y = y_2d * estimated_distance;
        
        RCLCPP_INFO(this->get_logger(), 
                    "Computed goal: x=%.3f, y=%.3f (distance=%.3f)", 
                    goal_x, goal_y, estimated_distance);
        
        // Publish goal position
        auto goal_msg = geometry_msgs::msg::Point();
        goal_msg.x = goal_x;
        goal_msg.y = goal_y;
        goal_msg.z = 0.0;  // Not used by navigation
        
        goal_publisher_->publish(goal_msg);
        RCLCPP_INFO(this->get_logger(), "Published goal position");
    }
    
    // ========== Utility Functions ==========
    
    double estimate_distance_from_energy(double energy) {
        // Energy decreases with distance (inverse square law for sound)
        // Map energy [0, 1] to distance [MAX_DISTANCE, MIN_DISTANCE]
        // Higher energy -> closer (smaller distance)
        // Lower energy -> farther (larger distance)
        
        // Clamp energy to [0, 1]
        energy = std::max(0.0, std::min(1.0, energy));
        
        // Linear interpolation: distance = MAX - (MAX - MIN) * energy
        double distance = MAX_DISTANCE - (MAX_DISTANCE - MIN_DISTANCE) * energy;
        
        // Alternative: inverse relationship for more realistic sound propagation
        // double distance = MIN_DISTANCE + (MAX_DISTANCE - MIN_DISTANCE) * (1.0 - energy);
        
        return distance;
    }
    
    // ========== Member Variables ==========
    
    rclcpp::Subscription<odas_ros_msgs::msg::OdasSsl>::SharedPtr ssl_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr goal_publisher_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SoundGoalPublisher>());
    rclcpp::shutdown();
    return 0;
}
