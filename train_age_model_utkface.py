import os
import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -- Konfigurasi lokasi dataset UTKFace --
# Dataset UTKFace biasanya memiliki nama file format: {age}_{gender}_{race}_{date}.jpg
# Contoh: “45_1_2_20170119204532533.jpg”
UTK_DIR = "data/UTKFace"

# Fungsi membaca dataset
def load_utkface_data(img_dir, target_size=(64, 64), max_samples=None):
    X = []
    y = []
    files = os.listdir(img_dir)
    if max_samples:
        files = files[:max_samples]
    for fname in files:
        try:
            parts = fname.split("_")
            age = float(parts[0])
        except Exception as e:
            # filename tidak sesuai format, skip
            continue
        fpath = os.path.join(img_dir, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # resize ke ukuran target
        img = cv2.resize(img, target_size)
        # normalisasi ke [0,1]
        img = img.astype("float32") / 255.0
        # expand dims ke (H, W, 1)
        img = np.expand_dims(img, axis=-1)
        X.append(img)
        y.append(age)
    X = np.array(X)
    y = np.array(y, dtype="float32")
    return X, y

def build_age_model(input_shape=(64,64,1)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='linear')
    ])
    return model

def main():
    print("[INFO] Memuat data UTKFace …")
    X, y = load_utkface_data(UTK_DIR, target_size=(64,64))
    print(f"[INFO] Total sampel: {len(X)}")

    # Bagi ke training & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Membangun model …")
    model = build_age_model(input_shape=(64,64,1))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        "src/umur_model/age_model.h5",
        monitor="val_loss", save_best_only=True, verbose=1
    )
    early = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

    print("[INFO] Memulai pelatihan …")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early]
    )

    print("[INFO] Pelatihan selesai.")

if __name__ == "__main__":
    main()
