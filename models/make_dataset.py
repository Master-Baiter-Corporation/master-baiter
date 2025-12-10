import os
import numpy as np
import librosa
import pickle
from pathlib import Path
from tqdm import tqdm

class AudioSpectrogramDataset:
    def __init__(self, root_dir, sample_rate=16000):
        """
        Args:
            root_dir: Path to directory containing class subfolders with WAV files
            sample_rate: Target sample rate (16000 Hz is common for audio classification)
        """
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        
        # Spectrogram parameters matching YAMNet
        self.n_mels = 64
        self.n_frames = 96
        self.fft_window_ms = 25  # milliseconds
        self.hop_length_ms = 10  # milliseconds
        self.fmin = 125  # Hz
        self.fmax = 7500  # Hz
        
        # Calculate FFT parameters
        self.n_fft = int(self.fft_window_ms * self.sample_rate / 1000)
        self.hop_length = int(self.hop_length_ms * self.sample_rate / 1000)
        
        # Data augmentation configuration per class
        self.augmentation_config = {
            'avance': 9,
            'droite': 9,
            'gauche': 9,
            'lucien': 9,
            'recule': 9,
            'unk': 11,
            'silence': 1
        }
    
    def augment_audio(self, y, augmentation_type):
        """
        Apply audio augmentation to create variations.
        
        Args:
            y: audio signal
            augmentation_type: type of augmentation to apply
        """
        if augmentation_type == 'pitch_shift_up':
            # Shift pitch up by 2 semitones
            return librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=2)
        
        elif augmentation_type == 'pitch_shift_down':
            # Shift pitch down by 2 semitones
            return librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=-2)
        
        elif augmentation_type == 'time_stretch_fast':
            # Speed up by 10%
            return librosa.effects.time_stretch(y, rate=1.1)
        
        elif augmentation_type == 'time_stretch_slow':
            # Slow down by 10%
            return librosa.effects.time_stretch(y, rate=0.9)
        
        elif augmentation_type == 'add_noise_light':
            # Add light white noise
            noise = np.random.normal(0, 0.005, len(y))
            return y + noise
        
        elif augmentation_type == 'add_noise_medium':
            # Add medium white noise
            noise = np.random.normal(0, 0.01, len(y))
            return y + noise
        
        elif augmentation_type == 'volume_up':
            # Increase volume by 20%
            return y * 1.2
        
        elif augmentation_type == 'volume_down':
            # Decrease volume by 20%
            return y * 0.8
        
        elif augmentation_type == 'time_shift':
            # Shift audio in time
            shift = np.random.randint(-self.sample_rate // 4, self.sample_rate // 4)
            return np.roll(y, shift)
        
        elif augmentation_type == 'pitch_shift_small_up':
            # Small pitch shift up
            return librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=1)
        
        elif augmentation_type == 'pitch_shift_small_down':
            # Small pitch shift down
            return librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=-1)
        
        elif augmentation_type == 'combined_1':
            # Combination: pitch + noise
            y_aug = librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=1)
            noise = np.random.normal(0, 0.005, len(y_aug))
            return y_aug + noise
        
        else:
            return y
    
    def audio_to_spectrogram(self, y):
        """Convert audio signal to mel spectrogram with shape (64, 96, 1)"""
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure exactly 96 frames
        if mel_spec_db.shape[1] >= self.n_frames:
            # Take first 96 frames if too long
            mel_spec_db = mel_spec_db[:, :self.n_frames]
        else:
            # Pad with zeros if too short
            pad_width = self.n_frames - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        
        # Reshape to (64, 96, 1)
        mel_spec_db = mel_spec_db.reshape(self.n_mels, self.n_frames, 1)
        
        return mel_spec_db
    
    def create_dataset(self):
        """
        Create dataset from folder structure with data augmentation.
        Returns: X (spectrograms), y (labels), label_names (class names)
        """
        spectrograms = []
        labels = []
        label_names = []
        
        # Define augmentation sequences for different amounts
        aug_sequences = {
            1: ['add_noise_light'],  # For silence (1 augmentation)
            9: ['pitch_shift_up', 'pitch_shift_down', 'time_stretch_fast', 
                'time_stretch_slow', 'add_noise_light', 'add_noise_medium',
                'volume_up', 'volume_down', 'time_shift'],  # For most classes
            11: ['pitch_shift_up', 'pitch_shift_down', 'time_stretch_fast', 
                 'time_stretch_slow', 'add_noise_light', 'add_noise_medium',
                 'volume_up', 'volume_down', 'time_shift', 
                 'pitch_shift_small_up', 'pitch_shift_small_down']  # For unk
        }
        
        # Get all class folders
        class_folders = sorted([d for d in Path(self.root_dir).iterdir() if d.is_dir()])
        
        print(f"Found {len(class_folders)} classes")
        
        # Process each class
        for class_idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            label_names.append(class_name)
            
            # Get all WAV files in this class
            audio_files = list(class_folder.glob('*.wav'))
            print(f"\nProcessing class '{class_name}': {len(audio_files)} files")
            
            # Determine number of augmentations for this class
            num_augmentations = self.augmentation_config.get(class_name, 0)
            
            if num_augmentations > 0:
                augmentation_types = aug_sequences[num_augmentations]
                print(f"  Augmentation: creating {num_augmentations} additional samples per file")
            else:
                print(f"  No augmentation for this class")
            
            class_spectrograms = []
            
            # Process each audio file
            for audio_file in tqdm(audio_files, desc=f"Class {class_name}"):
                try:
                    # Load audio file
                    y, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
                    
                    # Add original sample
                    spec = self.audio_to_spectrogram(y)
                    spectrograms.append(spec)
                    labels.append(class_idx)
                    
                    # Add augmented samples if needed
                    if num_augmentations > 0:
                        for aug_type in augmentation_types:
                            y_aug = self.augment_audio(y, aug_type)
                            spec_aug = self.audio_to_spectrogram(y_aug)
                            spectrograms.append(spec_aug)
                            labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
            
            original_count = len(audio_files)
            total_count = len([l for l in labels if l == class_idx])
            print(f"  Class '{class_name}': {original_count} original → {total_count} total samples")
        
        X = np.array(spectrograms)
        y = np.array(labels)
        
        print(f"\nTotal dataset size: {len(X)} samples")
        
        return X, y, label_names
    
    def normalize_variance_per_label(self, X, y):
        """
        Normalize each label's data to have the same variance.
        Adjusts magnitude so all classes have equal variance.
        """
        X_normalized = X.copy()
        unique_labels = np.unique(y)
        
        # Calculate target variance (mean variance across all labels)
        variances = []
        for label in unique_labels:
            mask = y == label
            label_data = X[mask]
            variances.append(np.var(label_data))
        
        target_variance = np.mean(variances)
        print(f"\nTarget variance: {target_variance:.4f}")
        
        # Normalize each label to target variance
        for label in unique_labels:
            mask = y == label
            label_data = X_normalized[mask]
            current_var = np.var(label_data)
            
            if current_var > 0:
                # Scale to match target variance
                scale_factor = np.sqrt(target_variance / current_var)
                X_normalized[mask] = label_data * scale_factor
                
                print(f"Label {label}: original var={current_var:.4f}, "
                      f"new var={np.var(X_normalized[mask]):.4f}, "
                      f"scale={scale_factor:.4f}")
        
        return X_normalized
    
    def save_dataset(self, X, y, label_names, output_path):
        """Save dataset to file"""
        dataset = {
            'X': X,  # Shape: (n_samples, 64, 96, 1)
            'y': y,  # Shape: (n_samples,)
            'label_names': label_names,  # List of class names
            'metadata': {
                'sample_rate': self.sample_rate,
                'n_mels': self.n_mels,
                'n_frames': self.n_frames,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'fmin': self.fmin,
                'fmax': self.fmax,
                'shape': X.shape
            }
        }
        
        # Save as numpy compressed file
        np.savez_compressed(output_path, **dataset)
        print(f"\nDataset saved to {output_path}")
        print(f"Shape: X={X.shape}, y={y.shape}")
        print(f"Classes: {label_names}")


def load_dataset(file_path):
    """
    Load the saved dataset.
    
    Returns:
        X: spectrograms array (n_samples, 64, 96, 1)
        y: labels array (n_samples,)
        label_names: list of class names
        metadata: dict with processing parameters
    """
    data = np.load(file_path, allow_pickle=True)
    
    X = data['X']
    y = data['y']
    label_names = data['label_names']
    metadata = data['metadata'].item() if 'metadata' in data else {}
    
    print(f"Loaded dataset: X shape={X.shape}, y shape={y.shape}")
    print(f"Classes: {label_names}")
    
    return X, y, label_names, metadata


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Configuration
    ROOT_DIR = "audios_cleans"  # Change this to your folder path
    OUTPUT_FILE = "audio_dataset_augmented.npz"  # Output filename
    
    # Create dataset
    print("Creating Audio Spectrogram Dataset with Data Augmentation...")
    dataset_creator = AudioSpectrogramDataset(root_dir=ROOT_DIR, sample_rate=16000)
    
    # Generate spectrograms (now with augmentation)
    print("\n" + "="*50)
    print("Step 1: Converting audio to spectrograms + Augmentation")
    print("="*50)
    X, y, label_names = dataset_creator.create_dataset()
    
    # Normalize variance per label
    print("\n" + "="*50)
    print("Step 2: Normalizing variance per label")
    print("="*50)
    X_normalized = dataset_creator.normalize_variance_per_label(X, y)
    
    # Save dataset
    print("\n" + "="*50)
    print("Step 3: Saving dataset")
    print("="*50)
    dataset_creator.save_dataset(X_normalized, y, label_names, OUTPUT_FILE)
    
    # Example: Load the dataset
    print("\n" + "="*50)
    print("Example: Loading the dataset")
    print("="*50)
    X_loaded, y_loaded, labels_loaded, metadata = load_dataset(OUTPUT_FILE)
    
    print("\nDataset ready for use!")
    print(f"First sample shape: {X_loaded[0].shape}")
    print(f"Total samples: {len(X_loaded)}")
    print(f"Labels distribution:")
    for idx, label_name in enumerate(labels_loaded):
        count = np.sum(y_loaded == idx)
        print(f"  {label_name}: {count} samples")
