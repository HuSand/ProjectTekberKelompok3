import numpy as np
import cv2
from tensorflow.keras.models import load_model as keras_load_model

# wajib ada
def load_model(model_path: str | None = None):
    """
    Muat model/weight umur.
    """
    # Jika tidak diberikan, gunakan model default (misal src/umur_model/age_model.h5)
    model_path = model_path or "src/umur_model/age_model.h5"
    try:
        model = keras_load_model(model_path)
        print(f"[INFO] Model umur dimuat dari {model_path}")
        return model
    except Exception as e:
        print(f"[WARNING] Tidak bisa memuat model: {e}")
        return None


# wajib ada
def predict_ages(gray_faces_64, model=None):
    """
    Param:
      gray_faces_64: List[np.ndarray (64,64) uint8]
      model: objek dari load_model()
    Return:
      List[float|int] -> usia per wajah, panjang = jumlah wajah
    """
    if len(gray_faces_64) == 0:
        return []

    ages = []

    # Jika tidak ada model, kembalikan nilai dummy
    if model is None:
        return [25.0 for _ in gray_faces_64]

    for face in gray_faces_64:
        # Preprocessing
        face_resized = cv2.resize(face, (64, 64))
        face_norm = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_norm, axis=(0, -1))  # (1,64,64,1)

        # Inferensi
        pred = model.predict(face_input, verbose=0)
        # Jika output 1 neuron (regresi umur)
        age = float(pred[0][0])
        # Clamp range agar realistis
        age = max(1.0, min(age, 90.0))
        ages.append(round(age, 1))

    return ages
