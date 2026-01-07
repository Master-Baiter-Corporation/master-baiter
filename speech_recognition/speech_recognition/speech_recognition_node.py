#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

try:
    import tensorflow as tf
except ImportError:
    tf = None


class SpeechRecognitionNode(Node):
    """
    ROS2 Node for speech recognition using TFLite model.
    
    Subscribes to:
        /audio_features (Float32MultiArray): Audio features input for the model
    
    Publishes to:
        /voice_command (String): Recognized command string
    """
    
    def __init__(self):
        super().__init__('speech_recognition_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('input_topic', '/audio_features')
        self.declare_parameter('output_topic', '/voice_command')
        self.declare_parameter('commands', ["avance", "droite", "gauche", "lucien", "recule", "silence", "unk"])
        self.declare_parameter('confidence_threshold', 0.8)
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.commands = self.get_parameter('commands').get_parameter_value().string_array_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        
        # Load TFLite model
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_model(model_path)
        
        # Create subscriber
        self.subscription = self.create_subscription(
            Float32MultiArray,
            input_topic,
            self.audio_callback,
            10
        )
        
        # Create publisher
        self.publisher = self.create_publisher(
            String,
            output_topic,
            10
        )
        
        self.get_logger().info(f'Speech Recognition Node initialized')
        self.get_logger().info(f'Subscribing to: {input_topic}')
        self.get_logger().info(f'Publishing to: {output_topic}')
        self.get_logger().info(f'Commands: {self.commands}')
        self.get_logger().info(f'Confidence threshold: {self.confidence_threshold}')
    
    def load_model(self, model_path):
        """
        Load the TFLite model.
        
        Args:
            model_path: Path to the .tflite model file
        """
        if not model_path:
            try:
                package_share_directory = get_package_share_directory('speech_recognition')
                model_path = os.path.join(package_share_directory, 'models', 'speech_model.tflite')
            except Exception as e:
                self.get_logger().error(f'Could not find default model path: {e}')
                return
        
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model file not found: {model_path}')
            return
        
        try:
            if tf is None:
                self.get_logger().error('TensorFlow is not installed. Please install: pip install tensorflow')
                return
            
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.get_logger().info(f'Model loaded successfully from: {model_path}')
            self.get_logger().info(f'Input shape: {self.input_details[0]["shape"]}')
            self.get_logger().info(f'Output shape: {self.output_details[0]["shape"]}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
    
    def preprocess_input(self, audio_data):
        """
        Preprocess audio data for the model.
        
        Args:
            audio_data: Raw audio features from the input topic
            
        Returns:
            Preprocessed numpy array ready for inference
        """
        input_array = np.array(audio_data, dtype=np.float32)
        
        # Get expected input shape from model
        expected_shape = self.input_details[0]['shape']
        
        if len(input_array.shape) == 1:
            if len(expected_shape) == 2:
                # Model expects [batch, features]
                input_array = input_array.reshape(1, -1)
            elif len(expected_shape) == 3:
                input_array = input_array.reshape(1, expected_shape[1], expected_shape[2])
        
        # Ensure the shape matches what the model expects
        if input_array.shape[1:] != tuple(expected_shape[1:]):
            self.get_logger().warning(
                f'Input shape {input_array.shape} does not match expected shape {expected_shape}'
            )
            # Pad or truncate as needed
            if np.prod(input_array.shape[1:]) < np.prod(expected_shape[1:]):
                # Pad with zeros
                pad_size = np.prod(expected_shape[1:]) - np.prod(input_array.shape[1:])
                input_array = np.pad(input_array.flatten(), (0, pad_size), 'constant')
            else:
                # Truncate
                input_array = input_array.flatten()[:np.prod(expected_shape[1:])]
            
            input_array = input_array.reshape(expected_shape)
        
        return input_array
    
    def audio_callback(self, msg):
        """
        Callback function for audio input.
        
        Args:
            msg: Float32MultiArray containing audio features
        """
        if self.interpreter is None:
            self.get_logger().warning('Model not loaded. Skipping inference.')
            return
        
        try:
            # Preprocess input
            input_data = self.preprocess_input(msg.data)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process output
            self.process_output(output_data)
            
        except Exception as e:
            self.get_logger().error(f'Error during inference: {e}')
    
    def process_output(self, output_data):
        """
        Process model output and publish recognized command.
        
        Args:
            output_data: Output from the TFLite model
        """
        # Get predicted class and confidence
        predictions = output_data[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Check if confidence is above threshold
        if confidence < self.confidence_threshold:
            self.get_logger().debug(
                f'Low confidence: {confidence:.2f} < {self.confidence_threshold}'
            )
            return
        
        # Get command name
        if predicted_class < len(self.commands):
            command = self.commands[predicted_class]
        else:
            self.get_logger().warning(
                f'Predicted class {predicted_class} is out of range for commands list'
            )
            return
        
        # Publish command
        msg = String()
        msg.data = command
        self.publisher.publish(msg)
        
        self.get_logger().info(
            f'Recognized command: "{command}" (confidence: {confidence:.2f})'
        )


def main(args=None):
    """Main function to initialize and spin the node."""
    rclpy.init(args=args)
    
    node = SpeechRecognitionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
