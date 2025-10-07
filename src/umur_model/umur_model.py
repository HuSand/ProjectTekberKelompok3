# src/umur_model/umur_model.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model as keras_load_model

def _infer_input_spec(model):
    ishape = model.input_shape
    if isinstance(ishape, (list, tuple)) and isinstance(ishape[0], (list, tuple)):
        ishape = ishape[0]
    # contoh: (None, 64, 64, 1) atau (None, 1, 64, 64)
    if len(ishape) == 4:
        # channels_last
        if ishape[-1] in (1,3):
            return (ishape[1], ishape[2], ishape[3]), True
        # channels_first
        if ishape[1] in (1,3):
            return (ishape[2], ishape[3], ishape[1]), False
    return (64,64,1), True

def load_model(model_path: str | None = None):
    model_path = model_path or "src/umur_model/age_model.h5"
    try:
        model = keras_load_model(model_path, compile=False)
        print(f"[INFO] Model umur dimuat dari {model_path}")
        return model
    except Exception as e:
        print(f"[WARNING] Tidak bisa memuat model: {e}")
        return None

def _prep_variants(gray, H, W, C):
    """
    Hasilkan beberapa varian preprocessing untuk dicoba.
    Return list of (np.ndarray ready_for_model, desc)
    """
    # pastikan single channel 2D
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    # resize
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    # equalize (optional)
    g_eq = cv2.equalizeHist(g)

    outs = []

    def _to_batch(x2d):
        if C == 1:
            x = x2d.astype(np.float32)
            x = x[..., None]  # (H,W,1)
        else:
            # nggak kejadian buat model kamu (C=1), tapi jaga-jaga
            x = cv2.cvtColor(x2d, cv2.COLOR_GRAY2RGB).astype(np.float32)  # (H,W,3)
        x = x[None, ...]  # (1,H,W,C)
        return x

    # 1) 0..1
    outs.append((_to_batch(g)/255.0, "norm01"))
    outs.append((_to_batch(g_eq)/255.0, "eq_norm01"))
    # 2) standardized (mean 127.5, std 128)
    outs.append((((_to_batch(g) - 127.5)/128.0), "std"))
    outs.append((((_to_batch(g_eq) - 127.5)/128.0), "eq_std"))
    # 3) raw 0..255
    outs.append((_to_batch(g), "raw"))
    outs.append((_to_batch(g_eq), "eq_raw"))

    return outs

def predict_ages(gray_faces_64, model=None):
    if len(gray_faces_64) == 0:
        return []

    # Kalau model None → dummy 25
    if model is None:
        return [25.0 for _ in gray_faces_64]

    ages = []
    for face in gray_faces_64:
        face_resized = cv2.resize(face, (64, 64))
        x = face_resized.astype("float32") / 255.0
        x = np.expand_dims(x, axis=(0, -1))  # (1,64,64,1)

        pred = model.predict(x, verbose=0)
        raw = float(pred[0][0])

        # ==== KALIBRASI SKALA ====
        # Banyak model regresi umur custom nge-output 0..1 (persentase umur 0..100).
        # Kita coba deteksi: kalau raw < 1.5 anggap dia skala 0..1 → kalikan 100.
        age = raw * 100.0 if raw <= 1.5 else raw

        # Opsional: affine tweak kecil biar gak “anak semua”
        # age = 0.85 * age + 3.0

        # Clamp biar realistis
        age = max(1.0, min(age, 90.0))
        ages.append(round(age, 1))

    return ages

