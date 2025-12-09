#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"

using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node {
public:
    MinimalSubscriber() : Node("minimal_subscriber") {
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "baiter_cmd", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
        
        cmd_vel_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10);
    }

private:
    void topic_callback(const std_msgs::msg::String& msg) const {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg.data.c_str());
        
        auto twist_msg = geometry_msgs::msg::Twist();
        
        if (msg.data == "avance") {
            twist_msg.linear.x = 0.5;  // Move forward
            twist_msg.angular.z = 0.0;
        }
        else if (msg.data == "recule") {
            twist_msg.linear.x = -0.5;  // Move backward
            twist_msg.angular.z = 0.0;
        }
        else if (msg.data == "gauche") {
            twist_msg.linear.x = 0.0;
            twist_msg.angular.z = 0.5;  // Turn left
        }
        else if (msg.data == "droite") {
            twist_msg.linear.x = 0.0;
            twist_msg.angular.z = -0.5;  // Turn right
        }
        else if (msg.data == "lucien") {
            twist_msg.linear.x = 0.0;  // Stop
            twist_msg.angular.z = 0.0;
        }
        else {
            RCLCPP_WARN(this->get_logger(), "Unknown command: '%s'", msg.data.c_str());
            return;
        }
        
        cmd_vel_publisher_->publish(twist_msg);
        RCLCPP_INFO(this->get_logger(), "Published cmd_vel");
    }
    
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_publisher_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MinimalSubscriber>());
    rclcpp::shutdown();
    return 0;
}
