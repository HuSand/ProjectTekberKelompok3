import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset FER-2013 dari Kaggle (ubah path sesuai letak dataset kamu)
train_dir = "data/fer2013/train"
val_dir   = "data/fer2013/test"

# Data generator (augmentasi data biar model lebih kuat)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

# Arsitektur CNN sederhana
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")  # 7 kelas emosi
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# Training
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen
)

# Simpan model
os.makedirs("src/emosi_model", exist_ok=True)
model.save("src/emosi_model/fer2013_cnn.h5")
print("✅ Model tersimpan di src/emosi_model/fer2013_cnn.h5")
