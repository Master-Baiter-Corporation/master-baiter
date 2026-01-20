from pathlib import Path
import numpy as np
import librosa
from tensorflow import keras
import tensorflow as tf
from difflib import SequenceMatcher
import argparse
import time

INFER_STATS = {"times": [], "warmup_left": 5}

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
    accuracy = matches / len(y_true) if len(y_true) > 0 else 0
    precision = matches / len(y_pred) if len(y_pred) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'matches': matches,
        'total_true': len(y_true),
        'total_pred': len(y_pred),
        'missing': len(y_true) - matches,
        'extra': len(y_pred) - matches
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

    # handle int8 quantization
    if input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        data = (data / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], data)

    t0 = time.perf_counter()
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    t1 = time.perf_counter()

    # collect stats
    if INFER_STATS["warmup_left"] > 0:
        INFER_STATS["warmup_left"] -= 1
    else:
        INFER_STATS["times"].append(t1 - t0)

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
            last_pred_idx = pred_idx

    probs_all = np.asarray(probs_all)
    avg_probs = np.mean(probs_all, axis=0)
    top = np.argsort(avg_probs)[::-1][:3]
    return avg_probs, top, triggers

def get_model_size(path):
    """Get model file size in KB"""
    return path.stat().st_size / 1024

def test_model(interpreter, name, wav_path, label_names, y_true, model_path):
    """Test a single model and return results"""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Taille: {get_model_size(model_path):.2f} KB")

    start = time.perf_counter()
    probs, top, triggers = predict_file(interpreter, wav_path, label_names, hop_s=0.05)
    end = time.perf_counter()
    times = INFER_STATS["times"]
    if len(times) > 0:
        avg_ms = (sum(times) / len(times)) * 1000
        print(f"  Inference only: {avg_ms:.3f} ms avg over {len(times)} calls (warmup skipped)")
    else:
        print("  Inference only: no timing samples collected")
    INFER_STATS["times"].clear()
    INFER_STATS["warmup_left"] = 5
    print(f"  Détections: {triggers}")

    results = sequence_accuracy(y_true, triggers)
    print(f"  Accuracy: {results['accuracy']:.2%} ({results['matches']}/{results['total_true']} détectés)")
    print(f"  Precision: {results['precision']:.2%} ({results['matches']}/{results['total_pred']} corrects)")
    print(f"  Missing: {results['missing']} | False: {results['extra']}")
    print(f"  Temps d'exécution: {end - start:.4f}s")

    return {
        'name': name,
        'size_kb': get_model_size(model_path),
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'time': end - start,
        'matches': results['matches'],
        'missing': results['missing'],
        'extra': results['extra']
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test audio file with all trained models")
    parser.add_argument("audio_file", nargs='?', default="a.wav",
                        help="Audio file to test (default: a.wav)")
    parser.add_argument("--ground-truth", nargs='+',
                        help="Ground truth labels (optional)")
    parser.add_argument("--auto", action='store_true',
                        help="Run all tests automatically without pausing")
    args = parser.parse_args()

    REPO = Path(__file__).resolve().parents[1]
    WAV_PATH = REPO / "data" / "sound_test" / args.audio_file
    MODELS_DIR = REPO / "models"
    DATASET_NPZ = REPO / "audio_dataset_augmented.npz"

    if not WAV_PATH.exists():
        raise FileNotFoundError(f"WAV not found: {WAV_PATH}")

    # Define all models to test
    models_to_test = [
        ("Modèle de base (TFLite)", MODELS_DIR / "yamnet_finetuned_model.tflite"),
        ("INT8 Quantization", MODELS_DIR / "model_int8.tflite"),
        ("FLOAT16 Quantization", MODELS_DIR / "model_float16.tflite"),
        ("Pruning 50%", MODELS_DIR / "model_pruned_50.tflite"),
        ("Pruning 70%", MODELS_DIR / "model_pruned_70.tflite"),
        ("Pruning 50% + INT8", MODELS_DIR / "model_pruned_50_int8.tflite"),
        ("Pruning 70% + INT8", MODELS_DIR / "model_pruned_70_int8.tflite"),
        ("Pruning 50% + FLOAT16", MODELS_DIR / "model_pruned_50_float16.tflite"),
    ]

    # Load label names
    if DATASET_NPZ.exists():
        label_names = load_label_names(DATASET_NPZ)
    else:
        label_names = ["avance", "droite", "gauche", "lucien", "recule", "silence", "unk"]

    # Ground truth
    y_true = args.ground_truth if args.ground_truth else [
        'avance', 'recule', 'droite', 'lucien', 'gauche',
        'droite', 'lucien', 'recule', 'avance', 'gauche',
        'avance', 'recule'
    ]

    print("\n" + "="*70)
    print("BENCHMARK DE TOUS LES MODÈLES")
    print("="*70)
    print(f"Fichier audio: {WAV_PATH.name}")
    print(f"Ground truth: {y_true}")
    print("="*70)

    # Test all models
    results = []
    for i, (name, model_path) in enumerate(models_to_test):
        if not model_path.exists():
            print(f"\nSKIP: {name} - Fichier non trouvé: {model_path.name}")
            continue

        try:
            interpreter = load_tflite(model_path)
            result = test_model(interpreter, name, WAV_PATH, label_names, y_true, model_path)
            results.append(result)

            # Pause between tests unless auto mode
            if not args.auto and i < len(models_to_test) - 1:
                input("\nAppuyez sur Entrée pour continuer...")

        except Exception as e:
            print(f"\nERROR testing {name}: {e}")

    if results:
        print("\n\n" + "="*90)
        print("TABLEAU RÉCAPITULATIF")
        print("="*90)
        print(f"{'Modèle':<35} {'Taille (KB)':<12} {'Accuracy':<10} {'Precision':<10} {'Temps (s)':<10}")
        print("-"*90)

        for r in results:
            print(f"{r['name']:<35} {r['size_kb']:>10.2f}   {r['accuracy']:>8.1%}   {r['precision']:>8.1%}   {r['time']:>8.4f}")

        print("="*90)

        print("\nMEILLEURS MODÈLES:")
        best_accuracy = max(results, key=lambda x: x['accuracy'])
        best_size = min(results, key=lambda x: x['size_kb'])
        best_speed = min(results, key=lambda x: x['time'])

        print(f"  Meilleure précision: {best_accuracy['name']} ({best_accuracy['accuracy']:.1%})")
        print(f"  Plus petit: {best_size['name']} ({best_size['size_kb']:.2f} KB)")
        print(f"  Plus rapide: {best_speed['name']} ({best_speed['time']:.4f}s)")
        print("="*90)
