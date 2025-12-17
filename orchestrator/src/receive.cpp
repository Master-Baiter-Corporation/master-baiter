#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"

#define COMMAND_DURATION_MS 3500

using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node {
public:
    MinimalSubscriber() : Node("minimal_subscriber") {
        cmd_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "baiter_cmd", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
        
        cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10);
        
        stop_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(COMMAND_DURATION_MS),
            std::bind(&MinimalSubscriber::stop_robot, this));
        
        stop_timer_->cancel();
    }

private:
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
            twist_msg.linear.x = 0.0;
            twist_msg.angular.z = 0.0;
        }
        else {
            RCLCPP_WARN(this->get_logger(), "Unknown command: '%s'", msg.data.c_str());
            return;
        }
        
        cmd_vel_publisher_->publish(twist_msg);
        RCLCPP_INFO(this->get_logger(), "Published cmd_vel");
        
        // Reset and restart the timer
        stop_timer_->reset();
    }
    
    void stop_robot() {
        auto twist_msg = geometry_msgs::msg::Twist();
        twist_msg.linear.x = 0.0;
        twist_msg.angular.z = 0.0;
        
        cmd_vel_publisher_->publish(twist_msg);
        RCLCPP_INFO(this->get_logger(), "Auto-stop: Published zero cmd_vel");
        
        // Cancel timer until next command
        stop_timer_->cancel();
    }
    
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cmd_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
    rclcpp::TimerBase::SharedPtr stop_timer_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}
