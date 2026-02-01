#!/usr/bin/env python3

from std_msgs.msg import String, UInt8MultiArray
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
import io
import wave
import vosk
import json

import rclpy
from rclpy.node import Node

vosk.SetLogLevel(-1)  # Réduire la verbosité des logs VOSK


class SpeechRecognitionNode(Node):
    """
    ROS2 Node for speech recognition using Vosk model.
    
    Subscribes to:
        /audio_features (UInt8MultiArray): Audio data in WAV format (as bytes)
    
    Publishes to:
        /voice_command (String): Recognized command string
    """
    
    def __init__(self):
        super().__init__('speech_recognition_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('input_topic', '/audio_features')
        self.declare_parameter('output_topic', '/voice_command')
        self.declare_parameter('commands', ["avance", "droite", "gauche", "lucien", "recule"])
        self.declare_parameter('confidence_threshold', 1.0)

        # Get parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.commands = self.get_parameter('commands').get_parameter_value().string_array_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        # Load Vosk model
        self.model = None
        self.load_model(self.model_path)
        
        # Deduplication: garder trace de la dernière commande détectée
        self.last_command = ""
        self.last_command_time = self.get_clock().now()
        self.dedup_window_sec = 1  # Ignorer les duplicatas pendant 1 seconde
        
        # Statistiques
        self.total_processed = 0
        self.total_recognized = 0
        
        # Create subscriber
        self.subscription = self.create_subscription(
            UInt8MultiArray,
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
        self.get_logger().info(f'Deduplication window: {self.dedup_window_sec}s')
    
    def load_model(self, model_path):
        """
        Load the Vosk model.
        
        Args:
            model_path: Path to the Vosk model directory
        """
        if not model_path:
            try:
                package_share_directory = get_package_share_directory('speech_recognition')
                model_path = os.path.join(package_share_directory, 'models', 'vosk-model-small-fr-0.22')
            except Exception as e:
                self.get_logger().error(f'Could not find default model path: {e}')
                return
        
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model directory not found: {model_path}')
            return
        
        try:
            self.model = vosk.Model(model_path)
            self.get_logger().info(f'Vosk model loaded successfully from: {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load Vosk model: {e}')
    
    def audio_callback(self, msg):
        """
        Callback function for audio input in WAV format.
        
        Args:
            msg: UInt8MultiArray containing WAV file data as bytes
        """
        if self.model is None:
            self.get_logger().warning('Model not loaded. Skipping inference.')
            return
        
        self.total_processed += 1
        
        try:
            # 1. Convertir UInt8MultiArray en bytes
            # msg.data contient déjà une liste d'entiers (0-255)
            wav_bytes = bytes(msg.data)
            
            if len(wav_bytes) < 100:
                self.get_logger().warning(f'WAV data too small: {len(wav_bytes)} bytes')
                return
            
            # 2. Créer un BytesIO pour simuler un fichier WAV
            byte_io = io.BytesIO(wav_bytes)
            
            # 3. Ouvrir le fichier WAV
            try:
                wf = wave.open(byte_io, 'rb')
            except Exception as e:
                self.get_logger().warning(f'Invalid WAV data received: {e}')
                return
            
            # 4. Vérifier les paramètres audio
            if wf.getnchannels() != 1:
                self.get_logger().warning(f'Expected mono audio, got {wf.getnchannels()} channels')
                wf.close()
                return
            
            if wf.getsampwidth() != 2:
                self.get_logger().warning(f'Expected 16-bit audio, got {wf.getsampwidth()*8}-bit')
                wf.close()
                return
            
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            duration_ms = (num_frames / sample_rate) * 1000
            
            # Log périodique
            if self.total_processed % 20 == 0:
                self.get_logger().debug(
                    f"Received Audio #{self.total_processed}: "
                    f"{sample_rate}Hz, {num_frames} frames, {duration_ms:.0f}ms"
                )
            
            # 5. Créer un nouveau recognizer pour ce chunk audio
            grammar = '["' + '", "'.join(self.commands) + '"]' # grammar limite les mots reconnus aux commandes attendues
            recognizer = vosk.KaldiRecognizer(self.model, sample_rate, grammar)
            recognizer.SetWords(True)  # Activer la sortie avec les timings et la confiance

            # 6. Traiter l'audio
            audio_data = wf.readframes(num_frames)
            
            # 7. Passer l'audio au recognizer
            recognizer.AcceptWaveform(audio_data) # AcceptWaveform traite l'audio par morceaux
            
            # 8. Obtenir le résultat final
            final_result = json.loads(recognizer.Result())

            # Fermer le fichier WAV
            wf.close()
            
            # 9. Traiter le résultat
            self.process_output(final_result)
            
        except Exception as e:
            self.get_logger().error(f'Error during audio processing: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def process_output(self, result_dict):
        """
        Process Vosk recognition result and publish recognized command.
        
        Args:
            result_dict: Dictionary from Vosk with 'text' field
        """
        # Extraire le texte reconnu
        text = result_dict.get('text', '').strip().lower()
        
        if not text: # Aucune parole détectée -> silence
            return
        
        if 'result' in result_dict and len(result_dict['result']) > 0:
            confidence = result_dict['result'][0].get('conf', 0.0)
        else:
            # Rejeter si pas de données de confiance
            self.get_logger().warning(f'No confidence data in result for: "{text}"')
            return
        
        # Rejeter si la confiance est trop faible
        if confidence < self.confidence_threshold:
            self.get_logger().info(f'Low confidence ({confidence:.2f}): "{text}" ignored')
            return

        # Déduplication: vérifier si c'est un duplicata récent
        current_time = self.get_clock().now()
        time_since_last = (current_time - self.last_command_time).nanoseconds / 1e9
        
        if text == self.last_command and time_since_last < self.dedup_window_sec: # C'est un duplicata, on ignore
            self.get_logger().debug(f'Duplicate command ignored: "{text}" ({time_since_last:.2f}s ago)')
            return
        
        # Mise à jour de l'historique
        self.last_command = text
        self.last_command_time = current_time
        self.total_recognized += 1
        
        # Publier la commande
        msg = String()
        msg.data = text
        self.publisher.publish(msg)
        
        self.get_logger().info(
            f'Recognized: "{text.upper()}" '
            f'(conf = {confidence:.2f}) '
            f'[{self.total_recognized}]'
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
        node.get_logger().info(
            f"Stopping node. "
            f"Processed: {node.total_processed}, "
            f"Recognized: {node.total_recognized}"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()