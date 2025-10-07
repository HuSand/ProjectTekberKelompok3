# src/emosi_model/emosi_model.py
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# --- bangun ulang model persis seperti config lama (nama layer penting!) ---
def build_fer2013_cnn(input_shape=(48, 48, 1), n_classes=7):
    L2 = regularizers.l2(0.01)  # di JSON conv2d_1 ada L2=0.01
    x = inp = keras.Input(shape=input_shape, name="input_1")

    # block 1
    x = layers.Conv2D(64, (3,3), padding="valid", activation="relu",
                      kernel_regularizer=L2, name="conv2d_1")(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu",
                      name="conv2d_2")(x)
    x = layers.BatchNormalization(name="batch_normalization_1")(x)
    x = layers.MaxPooling2D((2,2), name="max_pooling2d_1")(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)

    # block 2
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu",
                      name="conv2d_3")(x)
    x = layers.BatchNormalization(name="batch_normalization_2")(x)
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu",
                      name="conv2d_4")(x)
    x = layers.BatchNormalization(name="batch_normalization_3")(x)
    x = layers.MaxPooling2D((2,2), name="max_pooling2d_2")(x)
    x = layers.Dropout(0.5, name="dropout_2")(x)

    # block 3
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu",
                      name="conv2d_5")(x)
    x = layers.BatchNormalization(name="batch_normalization_4")(x)
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu",
                      name="conv2d_6")(x)
    x = layers.BatchNormalization(name="batch_normalization_5")(x)
    x = layers.MaxPooling2D((2,2), name="max_pooling2d_3")(x)
    x = layers.Dropout(0.5, name="dropout_3")(x)

    # block 4
    x = layers.Conv2D(512, (3,3), padding="same", activation="relu",
                      name="conv2d_7")(x)
    x = layers.BatchNormalization(name="batch_normalization_6")(x)
    x = layers.Conv2D(512, (3,3), padding="same", activation="relu",
                      name="conv2d_8")(x)
    x = layers.BatchNormalization(name="batch_normalization_7")(x)
    x = layers.MaxPooling2D((2,2), name="max_pooling2d_4")(x)
    x = layers.Dropout(0.5, name="dropout_4")(x)

    # head
    x = layers.Flatten(name="flatten_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.4, name="dropout_5")(x)
    x = layers.Dense(256, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.4, name="dropout_6")(x)
    x = layers.Dense(128, activation="relu", name="dense_3")(x)
    x = layers.Dropout(0.5, name="dropout_7")(x)
    out = layers.Dense(n_classes, activation="softmax", name="dense_4")(x)

    return keras.Model(inp, out, name="sequential_1")

class EmotionRecognizer:
    def __init__(self, weight_path: str = "src/emosi_model/fer2013_cnn.h5"):
        self.model = build_fer2013_cnn()
        # load weights by name biar cocok sama file lama
        self.model.load_weights(weight_path, by_name=True, skip_mismatch=False)
        self.emotions = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

        # preprocessing util
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def _prep(self, face_gray):
        if face_gray.ndim == 3:
            face_gray = cv2.cvtColor(face_gray, cv2.COLOR_BGR2GRAY)
        f = cv2.resize(face_gray, (48,48), interpolation=cv2.INTER_AREA)
        f = self._clahe.apply(f)
        f = f.astype("float32")/255.0
        f = f[...,None][None,...]  # (1,48,48,1)
        return f

    def predict_emotion(self, face_gray):
        x = self._prep(face_gray)
        p = self.model.predict(x, verbose=0)[0]  # (7,)
        idx = int(np.argmax(p))
        return self.emotions[idx], float(p[idx])
