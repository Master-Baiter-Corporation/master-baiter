import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class YAMNetTransferLearning:
    def __init__(self, backbone_path, num_classes, input_shape=(64, 96, 1)):
        """
        YAMNet-based model for audio classification with pre-trained backbone
        
        Args:
            backbone_path: Path to pre-trained backbone .h5 file
            num_classes: Number of output classes
            input_shape: Input spectrogram shape (64, 96, 1)
        """
        self.backbone_path = backbone_path
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.backbone = None
        self.history = None
        
    def load_pretrained_backbone(self):
        """Load pre-trained YAMNet backbone from .h5 file"""
        print(f"\nLoading pre-trained backbone from: {self.backbone_path}")
        try:
            self.backbone = keras.models.load_model(self.backbone_path, compile=False)
            print("✓ Backbone loaded successfully!")
            print(f"  Backbone input shape: {self.backbone.input_shape}")
            print(f"  Backbone output shape: {self.backbone.output_shape}")
            
            # Freeze all layers in the backbone
            self.backbone.trainable = False
            print("✓ All backbone layers frozen")
            
            # Count frozen parameters
            trainable_count = sum([tf.size(w).numpy() for w in self.backbone.trainable_weights])
            non_trainable_count = sum([tf.size(w).numpy() for w in self.backbone.non_trainable_weights])
            print(f"  Trainable parameters in backbone: {trainable_count:,}")
            print(f"  Non-trainable parameters in backbone: {non_trainable_count:,}")
            
            return self.backbone
            
        except Exception as e:
            print(f"✗ Error loading backbone: {e}")
            raise
    
    def build_model(self):
        """
        Build complete model with pre-trained YAMNet backbone + trainable classification head
        """
        if self.backbone is None:
            self.load_pretrained_backbone()
        
        # Build model with pre-trained backbone + new classification head
        inputs = keras.Input(shape=self.input_shape, name='input_spectrogram')
        
        # Pass through frozen backbone
        x = self.backbone(inputs)
        
        # Add trainable classification head
        x = layers.Dropout(0.5, name='dropout')(x)
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='classification_head'
        )(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='yamnet_transfer_learning')
        
        print("\n✓ Model built successfully!")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Count total parameters
        trainable_count = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_count = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        
        print("\n  Model compiled successfully!")
        print(f"  Total trainable parameters: {trainable_count:,}")
        print(f"  Total non-trainable parameters: {non_trainable_count:,}")
        print(f"  Optimizer: Adam (lr={learning_rate})")
        
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, callbacks=None):
        """
        Train the model (only the classification head)
        
        Args:
            X_train: Training spectrograms
            y_train: Training labels
            X_val: Validation spectrograms
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of Keras callbacks
        """
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    verbose=1,
                    min_lr=1e-7
                ),
                keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
        
        print("\n" + "="*60)
        print("Starting training (classification head only)...")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n Training completed!")
        return self.history
    
    def evaluate(self, X_test, y_test, label_names):
        """Evaluate model and print metrics"""
        print("\n" + "="*60)
        print("Evaluating model on test set...")
        print("="*60)
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate accuracy
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n Test Loss: {test_loss:.4f}")
        print(f" Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        for idx, label in enumerate(label_names):
            mask = y_test == idx
            if np.sum(mask) > 0:
                class_acc = np.sum((y_pred[mask] == y_test[mask])) / np.sum(mask)
                print(f"  {label}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        # Classification report
        print("\n" + "-"*60)
        print("Classification Report:")
        print("-"*60)
        print(classification_report(y_test, y_pred, target_names=label_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return y_pred, cm
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm, label_names, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names,
                    yticklabels=label_names,
                    cbar_kws={'label': 'Count'},
                    linewidths=0.5,
                    linecolor='gray')
        
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Confusion matrix saved as '{save_path}'")
        plt.show()
    
    def save_model(self, filepath='yamnet_trained_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f" Model saved to '{filepath}'")
    
    def save_tflite(self, filepath='yamnet_trained_model.tflite'):
        """Convert and save model as TFLite for deployment"""
        print(f"\nConverting model to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(filepath, 'wb') as f:
            f.write(tflite_model)
        
        # Print file size
        file_size = len(tflite_model) / (1024 * 1024)
        print(f" TFLite model saved to '{filepath}'")
        print(f"  Model size: {file_size:.2f} MB")


def load_dataset(file_path):
    """Load the saved dataset"""
    print(f"Loading dataset from: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    label_names = data['label_names'].tolist()
    print(f" Dataset loaded successfully")
    return X, y, label_names

if __name__ == "__main__":
    print("="*70)
    print(" " * 10 + "YAMNet Transfer Learning for Audio Event Detection")
    print(" " * 20 + "(Using Pre-trained Backbone)")
    print("="*70)
    
    # Configuration
    BACKBONE_PATH = "yamnet_1024_f32.h5"
    DATASET_PATH = "audio_dataset_augmented.npz"
    MODEL_SAVE_PATH = "yamnet_trained_model.h5"
    TFLITE_SAVE_PATH = "yamnet_trained_model.tflite"
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # Load dataset
    print("\n" + "="*70)
    print("STEP 1: Loading Dataset")
    print("="*70)
    X, y, label_names = load_dataset(DATASET_PATH)
    print(f"  Total samples: {len(X)}")
    print(f"  Input shape: {X.shape}")
    print(f"  Number of classes: {len(label_names)}")
    print(f"  Classes: {label_names}")
    
    # Print class distribution
    print("\n  Class distribution:")
    for idx, label in enumerate(label_names):
        count = np.sum(y == idx)
        percentage = (count / len(y)) * 100
        print(f"    {label}: {count} samples ({percentage:.1f}%)")
    
    # Split dataset (80/20)
    print("\n" + "="*70)
    print("STEP 2: Splitting Dataset (80% train, 20% test)")
    print("="*70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Verify split maintains class distribution
    print("\n  Training set class distribution:")
    for idx, label in enumerate(label_names):
        count = np.sum(y_train == idx)
        percentage = (count / len(y_train)) * 100
        print(f"    {label}: {count} samples ({percentage:.1f}%)")
    
    # Build model with pre-trained backbone
    print("\n" + "="*70)
    print("STEP 3: Building Model with Pre-trained Backbone")
    print("="*70)
    yamnet = YAMNetTransferLearning(
        backbone_path=BACKBONE_PATH,
        num_classes=len(label_names),
        input_shape=(64, 96, 1)
    )
    
    model = yamnet.build_model()
    
    print("\nModel architecture:")
    model.summary()
    
    # Compile model
    yamnet.compile_model(learning_rate=LEARNING_RATE)
    
    # Train model (only classification head)
    print("\n" + "="*70)
    print("STEP 4: Training Classification Head")
    print("="*70)
    history = yamnet.train(
        X_train, y_train,
        X_test, y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Step 5: Evaluate model
    print("\n" + "="*70)
    print("STEP 5: Final Evaluation")
    print("="*70)
    y_pred, cm = yamnet.evaluate(X_test, y_test, label_names)
    
    # Visualize results
    print("\n" + "="*70)
    print("STEP 6: Generating Visualizations")
    print("="*70)
    yamnet.plot_training_history()
    yamnet.plot_confusion_matrix(cm, label_names)
    
    # Save model
    print("\n" + "="*70)
    print("STEP 7: Saving Models")
    print("="*70)
    yamnet.save_model(MODEL_SAVE_PATH)
    yamnet.save_tflite(TFLITE_SAVE_PATH)
    
    # Final summary
    print("\n" + "="*70)
    print(" " * 25 + "TRAINING COMPLETED!")
    print("="*70)
    print(f"  {MODEL_SAVE_PATH} - Full Keras model")
    print(f"  {TFLITE_SAVE_PATH} - TFLite model for deployment")
    print(f"  best_model.h5 - Best model checkpoint")
    print(f"  confusion_matrix.png - Confusion matrix")
    print("="*70)
