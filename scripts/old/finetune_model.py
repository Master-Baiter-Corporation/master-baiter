import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

TRAINED_MODEL_PATH = "yamnet_trained_model.h5"  # may change
NEW_DATASET_PATH = "audio_bot_dataset_augmented.npz"
OUTPUT_MODEL_PATH = "yamnet_finetuned_model.h5"
OUTPUT_TFLITE_PATH = "yamnet_finetuned_model.tflite"

# Fine-tuning parameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001  # Lower LR for fine-tuning

print("="*70)
print(" " * 15 + "Fine-Tuning YAMNet Model (Full Network)")
print("="*70)

print("\n[1/6] Loading previously trained model...")
model = keras.models.load_model(TRAINED_MODEL_PATH)
print(f"✓ Model loaded from '{TRAINED_MODEL_PATH}'")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")

print("\n[2/6] Unfreezing all layers for fine-tuning...")
model.trainable = True
for layer in model.layers:
    layer.trainable = True

trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"All layers unfrozen")
print(f"Total trainable parameters: {trainable_count:,}")

print("\n[3/6] Recompiling model with lower learning rate...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"✓ Model recompiled with learning rate: {LEARNING_RATE}")

print("\n[4/6] Loading new dataset...")
data = np.load(NEW_DATASET_PATH, allow_pickle=True)
X = data['X']
y = data['y']
label_names = data['label_names'].tolist()

print(f"Dataset loaded from '{NEW_DATASET_PATH}'")
print(f"Total samples: {len(X)}")
print(f"Input shape: {X.shape}")

if X.shape[1:3] == (64, 96):
    print("\nDetected dimension mismatch - transposing data...")
    X = np.transpose(X, (0, 2, 1, 3))
    print(f"New shape after transpose: {X.shape}")

print(f"  Number of classes: {len(label_names)}")
print(f"  Classes: {label_names}")

# Class distribution
print("\nClass distribution:")
for idx, label in enumerate(label_names):
    count = np.sum(y == idx)
    percentage = (count / len(y)) * 100
    print(f"    {label}: {count} samples ({percentage:.1f}%)")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Training set: {len(X_train)} samples")
print(f"  Test set: {len(X_test)} samples")

print("\n[5/6] Fine-tuning entire model...")
print("="*70)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-8
    ),
    keras.callbacks.ModelCheckpoint(
        'best_finetuned_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\nFine-tuning completed!")

print("\n[6/6] Evaluating fine-tuned model...")
print("="*70)

# Predictions
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Overall accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

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

print("\nSaving fine-tuned models...")
model.save(OUTPUT_MODEL_PATH)
print(f"Model saved to '{OUTPUT_MODEL_PATH}'")

print(f"\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(OUTPUT_TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

file_size = len(tflite_model) / (1024 * 1024)
print(f"TFLite model saved to '{OUTPUT_TFLITE_PATH}'")
print(f"Model size: {file_size:.2f} MB")

print("\nGenerating training plots...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Fine-Tuning Accuracy', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Fine-Tuning Loss', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('finetuning_history.png', dpi=300, bbox_inches='tight')
print(f"Training history saved as 'finetuning_history.png'")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray')

plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix (Fine-Tuned)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('finetuned_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved as 'finetuned_confusion_matrix.png'")
plt.show()

print("\n" + "="*70)
print(" " * 20 + "FINE-TUNING COMPLETED!")
print("="*70)
print(f"  {OUTPUT_MODEL_PATH} - Fine-tuned Keras model")
print(f"  {OUTPUT_TFLITE_PATH} - Fine-tuned TFLite model")
print(f"  best_finetuned_model.h5 - Best checkpoint")
print(f"  finetuning_history.png - Training curves")
print(f"  finetuned_confusion_matrix.png - Confusion matrix")
print("="*70)
