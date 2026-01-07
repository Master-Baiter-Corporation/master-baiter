from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='orchestrator',
            executable='listener',
            name='listener_node'
        ),
        Node(
            package='orchestrator',
            executable='odas_listener',
            name='sound_goal_listerner_node'
        )
    ])
