from pathlib import Path
import numpy as np
import librosa
from tensorflow import keras

# ======= mêmes params que make_dataset.py =======
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

    # force (64, 96) sur l'axe temps
    if mel_db.shape[1] >= N_FRAMES:
        mel_db = mel_db[:, :N_FRAMES]
    else:
        mel_db = np.pad(mel_db, ((0, 0), (0, N_FRAMES - mel_db.shape[1])), mode="constant")

    # modèle attend (96, 64, 1)
    mel_db = mel_db.T  # (96, 64)
    return mel_db[..., None].astype(np.float32)  # (96, 64, 1)

def load_label_names(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    return data["label_names"].tolist()

def predict_file(model, wav_path: Path, label_names, window_s: float = 1.0, hop_s: float = 0.25):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)

    win = int(window_s * SR)         # 1.0s -> 16000
    hop = int(hop_s * SR)            # 0.25s -> 4000

    if win <= 0 or hop <= 0:
        raise ValueError("window_s et hop_s doivent être > 0")

    # fenêtres glissantes
    if len(y) < win:
        starts = [0]
    else:
        starts = list(range(0, len(y) - win + 1, hop))

    probs_all = []
    for k, start in enumerate(starts):
        chunk = y[start:start + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (0, win - len(chunk)))

        x = audio_to_spectrogram(chunk)[None, ...]  # (1,96,64,1)
        probs = model.predict(x, verbose=0)[0]
        probs_all.append(probs)

        pred_idx = int(np.argmax(probs))
        t0 = start / SR
        t1 = (start + win) / SR
        print(f"win {k:02d} [{t0:.2f}-{t1:.2f}s]: {label_names[pred_idx]} ({float(np.max(probs)):.3f})")

    probs_all = np.asarray(probs_all)

    # agrégation globale : moyenne sur toutes les fenêtres
    avg_probs = np.mean(probs_all, axis=0)

    top = np.argsort(avg_probs)[::-1][:3]
    return avg_probs, top

if __name__ == "__main__":
    REPO = Path(__file__).resolve().parents[1]  # remonte de /models vers la racine

    WAV_PATH = REPO / "data" / "sound_test" / "test_val.wav"
    MODEL_PATH = REPO / "models" / "yamnet_trained_model.h5"
    DATASET_NPZ = REPO / "audio_dataset_augmented.npz"  # optionnel

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not WAV_PATH.exists():
        raise FileNotFoundError(f"WAV not found: {WAV_PATH}")

    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model input:", model.input_shape, "| output:", model.output_shape)

    if DATASET_NPZ.exists():
        label_names = load_label_names(DATASET_NPZ)
    else:
        label_names = ["avance", "droite", "gauche", "lucien", "recule", "silence", "unk"]

    probs, top = predict_file(model, WAV_PATH, label_names, window_s=1.0, hop_s=0.25)

    print("\nTop-3 (global):")
    for idx in top:
        print(f"  {label_names[idx]}: {float(probs[idx]):.4f}")
