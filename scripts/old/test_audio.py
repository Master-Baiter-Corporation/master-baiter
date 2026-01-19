from pathlib import Path
import numpy as np
import librosa
from tensorflow import keras
import tensorflow as tf
from difflib import SequenceMatcher
import argparse
import time

SR = 16000
N_MELS = 64
N_FRAMES = 96
FFT_WINDOW_MS = 25
HOP_MS = 10
FMIN = 125
FMAX = 7500

N_FFT = int(FFT_WINDOW_MS * SR / 1000)
HOP_LENGTH = int(HOP_MS * SR / 1000)

def sequence_accuracy(y_true, y_pred):
    """Calculate accuracy accounting for insertions/deletions"""
    matcher = SequenceMatcher(None, y_true, y_pred)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    
    # Calculate metrics
    accuracy = matches / len(y_true)
    precision = matches / len(y_pred) if len(y_pred) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'matches': matches,
        'total_true': len(y_true),
        'total_pred': len(y_pred),
        'missing': len(y_true) - matches,  # Missed detections
        'extra': len(y_pred) - matches      # False detections
    }

def load_tflite(path):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter

def normalize_audio(y, target_db=-20.0, method='rms'):
    if method == 'rms':
        current_rms = np.sqrt(np.mean(y**2))
        if current_rms > 0:
            target_rms = 10**(target_db / 20.0)
            y = y * (target_rms / current_rms)
    
    elif method == 'peak':
        peak = np.abs(y).max()
        if peak > 0:
            target_peak = 10**(target_db / 20.0)
            y = y * (target_peak / peak)
    
    y = np.clip(y, -1.0, 1.0)
    
    return y

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

def load_label_names(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    return data["label_names"].tolist()

def run_tflite_predict(interpreter, data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Gestion de la quantization INT8
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        data = (data / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    return output

def predict_audio(interpreter, audio_chunk):
    probs = run_tflite_predict(interpreter, audio_chunk)[0]
    return probs

def predict_file(interpreter, wav_path: Path, label_names, window_s: float = 1.0, hop_s: float = 0.25):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    win = int(window_s * SR)
    hop = int(hop_s * SR)
    triggers = []
    last_pred_idx = 5

    if len(y) < win:
        starts = [0]
    else:
        starts = list(range(0, len(y) - win + 1, hop))

    probs_all = []
    for start in starts:
        chunk = y[start:start + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        chunk = normalize_audio(chunk, target_db=-20.0, method='rms')
        x = audio_to_spectrogram(chunk)[None, ...]  
        probs = predict_audio(interpreter, x)
        probs_all.append(probs)

        pred_idx = int(np.argmax(probs))
        if float(np.max(probs)) > 0.75:
            if (pred_idx!=last_pred_idx) and pred_idx<5:
                if len(triggers)>0:
                    if triggers[-1]!=label_names[pred_idx]: 
                        triggers.append(label_names[pred_idx])
                else:
                    triggers.append(label_names[pred_idx])
            #print(f"{label_names[pred_idx]} ({float(np.max(probs)):.3f})")
            last_pred_idx = pred_idx

    probs_all = np.asarray(probs_all)
    avg_probs = np.mean(probs_all, axis=0)
    top = np.argsort(avg_probs)[::-1][:3]
    return avg_probs, top, triggers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test audio file with trained models")
    parser.add_argument("audio_file", nargs='?', default="a.wav", 
                        help="Audio file to test (default: a.wav)")
    parser.add_argument("--ground-truth", nargs='+', 
                        help="Ground truth labels (optional)")
    args = parser.parse_args()

    REPO = Path(__file__).resolve().parents[1]

    WAV_PATH = REPO / "data" / "sound_test" / "a.wav"
    MODEL_PRUNED_PATH = REPO / "models" / "model_pruned.tflite"
    MODEL_QUANTIZED_PATH = REPO / "models" / "model_full_int8.tflite"
    MODEL_PATH = REPO / "models" / "yamnet_finetuned_model.tflite"
    DATASET_NPZ = REPO / "audio_dataset_augmented.npz"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not WAV_PATH.exists():
        raise FileNotFoundError(f"WAV not found: {WAV_PATH}")

    interpreter_base = load_tflite(MODEL_PATH)
    interpreter_pruned = load_tflite(MODEL_PRUNED_PATH)
    interpreter_quant = load_tflite(MODEL_QUANTIZED_PATH)

    if DATASET_NPZ.exists():
        label_names = load_label_names(DATASET_NPZ)
    else:
        label_names = ["avance", "droite", "gauche", "lucien", "recule", "silence", "unk"]

    y_true = args.ground_truth if args.ground_truth else ['avance', 'recule', 'droite', 'lucien', 'gauche', 'droite', 'lucien', 'recule', 'avance', 'gauche', 'avance', 'recule']
    print(f"Hardcoded true labels: {y_true}")

    print("\n--- Modèle de base ---")
    start = time.perf_counter()
    probs, top, triggers = predict_file(interpreter_base, WAV_PATH, label_names, hop_s=0.05)
    end = time.perf_counter()
    print(triggers)
    results = sequence_accuracy(y_true, triggers)
    print(f"  Accuracy: {results['accuracy']:.2%} ({results['matches']}/{results['total_true']} detected)")
    print(f"  Precision: {results['precision']:.2%} ({results['matches']}/{results['total_pred']} correct)")
    print(f"  Missing detections: {results['missing']}")
    print(f"  False detections: {results['extra']}")
    print(f"Temps d'exécution : {end - start:.4f} secondes")


    input("\nAppuyez sur Entrée pour le modèle PRUNED...")
    start = time.perf_counter()
    probs, top, triggers = predict_file(interpreter_pruned, WAV_PATH, label_names, hop_s=0.05)
    end = time.perf_counter()
    print(triggers)
    results = sequence_accuracy(y_true, triggers)
    print(f"  Accuracy: {results['accuracy']:.2%} ({results['matches']}/{results['total_true']} detected)")
    print(f"  Precision: {results['precision']:.2%} ({results['matches']}/{results['total_pred']} correct)")
    print(f"  Missing detections: {results['missing']}")
    print(f"  False detections: {results['extra']}")
    print(f"Temps d'exécution : {end - start:.4f} secondes")

    input("\nAppuyez sur Entrée pour le modèle QUANTIZED...")
    start = time.perf_counter()
    probs, top, triggers = predict_file(interpreter_quant, WAV_PATH, label_names, hop_s=0.05)
    end = time.perf_counter()
    print(triggers)
    results = sequence_accuracy(y_true, triggers)
    print(f"  Accuracy: {results['accuracy']:.2%} ({results['matches']}/{results['total_true']} detected)")
    print(f"  Precision: {results['precision']:.2%} ({results['matches']}/{results['total_pred']} correct)")
    print(f"  Missing detections: {results['missing']}")
    print(f"  False detections: {results['extra']}")
    print(f"Temps d'exécution : {end - start:.4f} secondes")
