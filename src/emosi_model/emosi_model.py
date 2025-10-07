# src/emosi_model/emosi_model.py
# Build ulang arsitektur FER2013 (48x48x1 → 7 kelas) lalu load_weights dari .h5
# Hindari deserialisasi JSON lama Keras 2.x yang bikin error di Keras 3.

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

EMO_CLASSES = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def _build_fer2013_model(input_shape=(48,48,1), num_classes=7):
    x = inp = layers.Input(shape=input_shape, name="input_1")

    # Block 1 (sesuai JSON: Conv(64, valid) -> Conv(64, same) -> BN -> MaxPool -> Dropout 0.5)
    x = layers.Conv2D(64, (3,3), padding="valid", activation="relu", name="conv2d_1")(x)
    x = layers.Conv2D(64, (3,3), padding="same",  activation="relu", name="conv2d_2")(x)
    x = layers.BatchNormalization(name="batch_normalization_1")(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name="max_pooling2d_1")(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)

    # Block 2: Conv(128) x2 + BN -> MaxPool -> Dropout 0.5
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu", name="conv2d_3")(x)
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu", name="conv2d_4")(x)
    x = layers.BatchNormalization(name="batch_normalization_2")(x)
    x = layers.BatchNormalization(name="batch_normalization_3")(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name="max_pooling2d_2")(x)
    x = layers.Dropout(0.5, name="dropout_2")(x)

    # Block 3: Conv(256) x2 + BN -> MaxPool -> Dropout 0.5
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu", name="conv2d_5")(x)
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu", name="conv2d_6")(x)
    x = layers.BatchNormalization(name="batch_normalization_4")(x)
    x = layers.BatchNormalization(name="batch_normalization_5")(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name="max_pooling2d_3")(x)
    x = layers.Dropout(0.5, name="dropout_3")(x)

    # Block 4: Conv(512) x2 + BN -> MaxPool -> Dropout 0.5
    x = layers.Conv2D(512, (3,3), padding="same", activation="relu", name="conv2d_7")(x)
    x = layers.Conv2D(512, (3,3), padding="same", activation="relu", name="conv2d_8")(x)
    x = layers.BatchNormalization(name="batch_normalization_6")(x)
    x = layers.BatchNormalization(name="batch_normalization_7")(x)
    x = layers.MaxPooling2D(pool_size=(2,2), name="max_pooling2d_4")(x)
    x = layers.Dropout(0.5, name="dropout_4")(x)

    x = layers.Flatten(name="flatten_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.4, name="dropout_5")(x)
    x = layers.Dense(256, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.4, name="dropout_6")(x)
    x = layers.Dense(128, activation="relu", name="dense_3")(x)
    x = layers.Dropout(0.5, name="dropout_7")(x)
    out = layers.Dense(num_classes, activation="softmax", name="dense_4")(x)

    return models.Model(inp, out, name="fer2013_cnn_rebuilt")

class EmotionRecognizer:
    def __init__(self, h5_path: str = "src/emosi_model/fer2013_cnn.h5", json_path: str | None = None):
        """
        Bangun arsitektur yang kompatibel dengan file weights FER2013 lama, lalu load weights.
        Abaikan json_path: tidak dipakai karena Keras 3 susah deserialisasi JSON 2.x.
        """
        wfile = Path(h5_path)
        if not wfile.exists():
            raise FileNotFoundError(f"Model weights (.h5) tidak ditemukan: {wfile}")

        self.model = _build_fer2013_model(input_shape=(48,48,1), num_classes=len(EMO_CLASSES))
        # load_weights (bukan load_model)
        self.model.load_weights(str(wfile))
        self.emotions = EMO_CLASSES

        # Preproc util
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def _preprocess(self, img_gray):
        # pastikan grayscale
        if img_gray.ndim == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
        f = cv2.resize(img_gray, (48,48), interpolation=cv2.INTER_AREA)
        f = self._clahe.apply(f)
        f = f.astype("float32")/255.0
        f = f[...,None][None,...]  # (1,48,48,1)
        return f

    def predict_emotion(self, face_img, unknown_threshold: float = 0.38):
        """
        face_img: BGR atau GRAY crop wajah (apa saja), kita urus preprocess-nya.
        unknown_threshold: kalau max prob < threshold, balikin 'Unknown'.
        return: (label, prob_max)
        """
        x = self._preprocess(face_img)
        probs = self.model.predict(x, verbose=0)[0]
        k = int(np.argmax(probs))
        p = float(probs[k])
        lbl = self.emotions[k] if p >= unknown_threshold else "Unknown"
        return lbl, p
