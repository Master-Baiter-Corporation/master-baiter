from pathlib import Path
import numpy as np
import librosa
# On remplace tensorflow par tflite_runtime pour plus de légèreté
import tflite_runtime.interpreter as tflite
from difflib import SequenceMatcher
import argparse
import time

# --- Constantes Audio ---
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
    matcher = SequenceMatcher(None, y_true, y_pred)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    return {
        'accuracy': matches / len(y_true),
        'precision': matches / len(y_pred) if len(y_pred) > 0 else 0,
        'matches': matches,
        'total_true': len(y_true),
        'total_pred': len(y_pred),
        'missing': len(y_true) - matches,
        'extra': len(y_pred) - matches
    }

def load_coral_tflite(path):
    """Charge le modèle spécifiquement pour la Google Coral"""
    try:
        # On charge le délégué Edge TPU (libedgetpu.so.1 sous Linux/Raspberry Pi)
        interpreter = tflite.Interpreter(
            model_path=str(path),
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        interpreter.allocate_tensors()
        print("✅ Modèle chargé avec succès sur l'accélérateur Coral !")
        return interpreter
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la Coral : {e}")
        print("Vérifiez que la clé est branchée et que libedgetpu1-std est installé.")
        exit(1)

def normalize_audio(y, target_db=-20.0, method='rms'):
    current_rms = np.sqrt(np.mean(y**2))
    if current_rms > 0:
        target_rms = 10**(target_db / 20.0)
        y = y * (target_rms / current_rms)
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

def run_tflite_predict(interpreter, data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Le modèle Coral est OBLIGATOIREMENT INT8
    scale, zero_point = input_details[0]['quantization']
    data_int8 = (data / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], data_int8)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])

    # Dequantization pour récupérer les probabilités en float
    scale_out, zero_out = output_details[0]['quantization']
    output_float = (output.astype(np.float32) - zero_out) * scale_out
    return output_float

def predict_file(interpreter, wav_path: Path, label_names, window_s: float = 1.0, hop_s: float = 0.25):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    win = int(window_s * SR)
    hop = int(hop_s * SR)
    triggers = []
    last_pred_idx = 5

    starts = list(range(0, len(y) - win + 1, hop)) if len(y) >= win else [0]

    probs_all = []
    for start in starts:
        chunk = y[start:start + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))
        chunk = normalize_audio(chunk)
        x = audio_to_spectrogram(chunk)[None, ...]  
        
        # Inférence via la Coral
        probs = run_tflite_predict(interpreter, x)[0]
        probs_all.append(probs)

        pred_idx = int(np.argmax(probs))
        if float(np.max(probs)) > 0.75:
            if (pred_idx != last_pred_idx) and pred_idx < 5:
                if len(triggers) == 0 or triggers[-1] != label_names[pred_idx]:
                    triggers.append(label_names[pred_idx])
            last_pred_idx = pred_idx

    return np.mean(probs_all, axis=0), triggers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", nargs='?', default="a.wav")
    args = parser.parse_args()

    REPO = Path(__file__).resolve().parents[1]
    WAV_PATH = REPO / "data" / "sound_test" / args.audio_file
    # SEUL le modèle compilé pour Edge TPU peut être utilisé ici
    MODEL_QUANT_PATH = REPO / "models" / "model_full_int8_edgetpu.tflite"

    # Chargement de l'interpréteur avec délégué Coral
    interpreter = load_coral_tflite(MODEL_QUANT_PATH)

    label_names = ["avance", "droite", "gauche", "lucien", "recule", "silence", "unk"]
    y_true = ['avance', 'recule', 'droite', 'lucien', 'gauche', 'droite', 'lucien', 'recule', 'avance', 'gauche', 'avance', 'recule']

    print(f"\n--- Inférence sur Google Coral (Edge TPU) ---")
    start = time.perf_counter()
    _, triggers = predict_file(interpreter, WAV_PATH, label_names, hop_s=0.05)
    end = time.perf_counter()

    print(f"Triggers détectés : {triggers}")
    res = sequence_accuracy(y_true, triggers)
    print(f"Accuracy : {res['accuracy']:.2%} | Temps : {end - start:.4f}s")
