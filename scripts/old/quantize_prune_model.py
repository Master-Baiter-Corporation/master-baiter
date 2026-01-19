import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import librosa
from pathlib import Path

SR = 16000
N_MELS = 64
N_FRAMES = 96
FFT_WINDOW_MS = 25
HOP_MS = 10
FMIN = 125
FMAX = 7500

N_FFT = int(FFT_WINDOW_MS * SR / 1000)
HOP_LENGTH = int(HOP_MS * SR / 1000)

def audio_to_spectrogram(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] >= N_FRAMES:
        mel_db = mel_db[:, :N_FRAMES]
    else:
        mel_db = np.pad(mel_db, ((0, 0), (0, N_FRAMES - mel_db.shape[1])), mode="constant")

    mel_db = mel_db.T
    return mel_db[..., None].astype(np.float32)

model = tf.keras.models.load_model('yamnet_finetuned_model.h5', compile=False)
window_s = 1.0
hop_s = 0.05
REPO = Path(__file__).resolve().parents[1]
wav_path = REPO / "data" / "sound_test" / "test_val.wav"
x_train = []
y, _ = librosa.load(wav_path, sr=SR, mono=True)
win = int(window_s * SR)
hop = int(hop_s * SR)

if win <= 0 or hop <= 0:
    raise ValueError("window_s et hop_s doivent être > 0")

if len(y) < win:
    starts = [0]
else:
    starts = list(range(0, len(y) - win + 1, hop))

probs_all = []
for k, start in enumerate(starts):
    chunk = y[start:start + win]
    if len(chunk) < win:
        chunk = np.pad(chunk, (0, win - len(chunk)))

    x_train.append(audio_to_spectrogram(chunk)[None, ...])

x_train = np.array(x_train)
x_train = np.squeeze(x_train, axis=1)
print(x_train.shape)

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        yield [tf.cast(input_value, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open('model_full_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)

model = tf.keras.models.load_model('yamnet_finetuned_model.h5', compile=False)
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

model_for_pruning = prune_low_magnitude(model, 
    pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.5, begin_step=0))

model_final = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

converter = tf.lite.TFLiteConverter.from_keras_model(model_final)
converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]

tflite_pruned_model = converter.convert()

with open('model_pruned.tflite', 'wb') as f:
    f.write(tflite_pruned_model)