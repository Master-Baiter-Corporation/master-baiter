import rclpy
from rclpy.node import Node
import numpy as np
import scipy.signal
import io
import wave

# Messages ROS
# On utilise UInt8MultiArray car un fichier .wav est une suite d'octets (binaire)
from std_msgs.msg import UInt8MultiArray

# Import conditionnel du message AudioFrame
try:
    from audio_utils_msgs.msg import AudioFrame
except ImportError:
    print("Erreur: Impossible d'importer AudioFrame. Vérifiez 'audio_utils_msgs'.")


class AudioBridgeOdasToVosk(Node):
    def __init__(self):
        super().__init__('audio_bridge_odas_vosk')
        
        # --- Paramètres Audio ---
        self.src_sr = 44100       # Fréquence source (ODAS /sss)
        self.target_sr = 16000    # Fréquence cible (Vosk)
        self.n_channels = 4       # Nombre de sources suivies par ODAS
        
        # Fenêtre glissante : 960 ms (suffisant pour la reconnaissance vocale)
        self.window_duration = 0.96  # secondes
        self.buffer_len = int(self.src_sr * self.window_duration)
        
        # Intervalle de publication : 50 ms (20 Hz)
        self.publish_interval = 0.05  # secondes
        
        # Buffer circulaire unique pour la source active
        self.audio_buffer = np.zeros(self.buffer_len, dtype=np.float32)
        
        # --- Subscriber (Entrée) ---
        self.subscription = self.create_subscription(
            AudioFrame, 
            '/sss', 
            self.listener_callback, 
            10
        )
        
        # --- Publisher (Sortie) ---
        # Publie des octets (le fichier wav complet avec header)
        self.feature_pub = self.create_publisher(UInt8MultiArray, '/audio_features', 10)
        
        # Gestion du timing pour la publication
        self.samples_since_last_publish = 0
        self.samples_threshold = int(self.src_sr * self.publish_interval)
        
        # Statistiques (optionnel)
        self.total_packets_received = 0
        self.total_wavs_published = 0
        
        self.get_logger().info("Audio Bridge ODAS -> Vosk Node Started")
        self.get_logger().info(f"Window duration: {self.window_duration * 1000:.0f} ms")
        self.get_logger().info(f"Publishing interval: {self.publish_interval * 1000:.0f} ms ({1/self.publish_interval:.0f} Hz)")
        self.get_logger().info(f"Source: {self.src_sr} Hz -> Target: {self.target_sr} Hz")
    
    def listener_callback(self, msg):
        """
        Reçoit l'audio brut (4 canaux entrelacés), sélectionne le meilleur canal,
        met à jour le buffer circulaire et publie quand nécessaire.
        """
        self.total_packets_received += 1
        
        # 1. Conversion Bytes -> Float32 normalisé
        # ODAS envoie du int16 (signed_16)
        try:
            raw_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Error while converting raw data: {e}")
            return
            
        raw_data = raw_data / 32768.0  # Normalisation entre -1.0 et 1.0
        
        # 2. Désentrelacement des canaux (Interleaved -> Planar)
        try:
            chunk_per_channel = raw_data.reshape(-1, self.n_channels)
        except ValueError as e:
            self.get_logger().warn(f"Invalid packet size received: {len(raw_data)} samples, error: {e}")
            return
        
        # 3. Sélection du canal actif (Celui avec le plus d'énergie dans ce chunk)
        # Calcul de l'énergie (RMS) de chaque canal sur ce paquet
        energies = np.sum(np.abs(chunk_per_channel), axis=0)
        active_channel_idx = np.argmax(energies)
        active_samples = chunk_per_channel[:, active_channel_idx]
        
        # Log occasionnel pour debug (tous les 100 paquets)
        if self.total_packets_received % 100 == 0:
            max_energy = energies[active_channel_idx]
            self.get_logger().debug(
                f"Packet {self.total_packets_received}: "
                f"Active channel={active_channel_idx}, "
                f"Energy={max_energy:.2f}, "
                f"Samples={len(active_samples)}"
            )
        
        # 4. Mise à jour du Buffer Circulaire (fenêtre glissante)
        num_new_samples = len(active_samples)
        self.audio_buffer = np.roll(self.audio_buffer, -num_new_samples)
        self.audio_buffer[-num_new_samples:] = active_samples
        
        # 5. Vérification du timing pour publication
        self.samples_since_last_publish += num_new_samples
        
        if self.samples_since_last_publish >= self.samples_threshold:
            self.process_and_publish()
            self.samples_since_last_publish = 0
    
    def process_and_publish(self):
        """
        Récupère le buffer, ré-échantillonne, convertit en WAV et publie.
        """
        # A. Vérification : le buffer contient-il du signal ?
        buffer_energy = np.sum(np.abs(self.audio_buffer))
        
        # Si le buffer est quasiment vide (silence), on peut choisir de ne pas publier
        # Seuil arbitraire : si l'énergie totale < 0.01, c'est probablement du silence
        if buffer_energy < 0.01:
            self.get_logger().debug("Buffer contains mainly silence, publication ignored")
            return
        
        # B. Ré-échantillonnage (44100 Hz -> 16000 Hz)
        num_samples_target = int(len(self.audio_buffer) * self.target_sr / self.src_sr)
        y_16k = scipy.signal.resample(self.audio_buffer, num_samples_target)
        
        # C. Conversion Float32 -> Int16 (Format requis pour WAV PCM)
        # Clipping pour éviter les débordements au-delà de [-1.0, 1.0]
        y_16k = np.clip(y_16k, -1.0, 1.0)
        audio_int16 = (y_16k * 32767).astype(np.int16)
        
        # D. Création du fichier WAV en mémoire
        byte_io = io.BytesIO()
        try:
            with wave.open(byte_io, 'wb') as wf:
                wf.setnchannels(1)                  # Mono
                wf.setsampwidth(2)                  # 2 octets (16 bits)
                wf.setframerate(self.target_sr)     # 16000 Hz
                wf.writeframes(audio_int16.tobytes())
        except Exception as e:
            self.get_logger().error(f"Error while creating WAV file: {e}")
            return
        
        # Récupérer les octets du fichier complet (Header + Data)
        wav_bytes = byte_io.getvalue()
        
        # E. Publication (UInt8MultiArray)
        msg = UInt8MultiArray()
        msg.data = list(wav_bytes)  # Conversion bytes -> list d'entiers
        
        self.feature_pub.publish(msg)
        self.total_wavs_published += 1
        
        # Log périodique
        if self.total_wavs_published % 20 == 0:  # Toutes les secondes (20 Hz)
            self.get_logger().info(
                f"Published {self.total_wavs_published} WAV | "
                f"Size: {len(wav_bytes)} bytes | "
                f"Duration: {len(audio_int16)/self.target_sr*1000:.0f} ms | "
                f"Energy: {buffer_energy:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = AudioBridgeOdasToVosk()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Stopping node. Total packets received: {node.total_packets_received}, WAV published: {node.total_wavs_published}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()