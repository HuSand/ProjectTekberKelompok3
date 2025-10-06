import cv2
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

class EmotionRecognizer:
    def __init__(self, model_path: str = "src/emosi_model/fer2013_cnn.h5"):
        """
        Inisialisasi model emosi.
        model_path -> path ke file model .h5 yang sudah dilatih (FER2013 atau dataset lain).
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file tidak ditemukan: {model_file}")
        
        self.model = load_model(str(model_file))
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def preprocess_face(self, face_gray, target_size=(48,48)):
        """
        Resize dan normalisasi wajah grayscale agar sesuai input model.
        """
        face = cv2.resize(face_gray, target_size)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)   # channel dim
        face = np.expand_dims(face, axis=0)    # batch dim
        return face

    def predict_emotion(self, face_gray):
        """
        Prediksi emosi dari 1 wajah grayscale.
        Return: label string (misal: "Happy").
        """
        processed = self.preprocess_face(face_gray)
        preds = self.model.predict(processed, verbose=0)[0]
        emotion_idx = np.argmax(preds)
        return self.emotions[emotion_idx], float(np.max(preds))