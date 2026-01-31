from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for speech recognition node."""
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description='Path to the Vosk model directory.'
    )
    
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/audio_features',
        description='Topic name for audio input features'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/voice_command',
        description='Topic name for recognized voice commands'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='1.0',
        description='Minimum confidence threshold for command recognition'
    )
    
    # Define the node
    speech_recognition_node = Node(
        package='speech_recognition',
        executable='speech_recognition_node',
        name='speech_recognition_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'input_topic': LaunchConfiguration('input_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
            'commands': ["avance", "droite", "gauche", "lucien", "recule"],
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
        }],
        emulate_tty=True,
    )
    
    return LaunchDescription([
        model_path_arg,
        input_topic_arg,
        output_topic_arg,
        confidence_threshold_arg,
        speech_recognition_node,
    ])
