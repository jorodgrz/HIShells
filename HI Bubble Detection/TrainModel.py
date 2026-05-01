# train_unet_shells.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
import os

# === CONFIGURATION ===
INPUT_SIZE = (64, 64)
NUM_SAMPLES = 500
EPOCHS = 10
BATCH_SIZE = 8
OUTPUT_MODEL_PATH = "models/unet_shell_segmentation.h5"

# === GENERATE SYNTHETIC DATASET ===
def generate_synthetic_shell(size=64, with_shell=True):
    image = np.random.rand(size, size) * 0.3  # background noise
    mask = np.zeros((size, size), dtype=np.uint8)

    if with_shell:
        for _ in range(np.random.randint(1, 3)):
            x, y = np.random.randint(16, size - 16, size=2)
            r = np.random.randint(5, 10)
            cv2.circle(image, (x, y), r, 0.05, thickness=2)
            cv2.circle(mask, (x, y), r, 1, thickness=-1)

    image = np.clip(image, 0, 1)
    return image[..., np.newaxis], mask[..., np.newaxis]

# === BUILD U-NET ===
def build_unet(input_shape=(64, 64, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)

    b = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)

    u2 = layers.UpSampling2D()(b)
    concat2 = layers.Concatenate()([u2, c2])
    c3 = layers.Conv2D(32, 3, activation='relu', padding='same')(concat2)

    u1 = layers.UpSampling2D()(c3)
    concat1 = layers.Concatenate()([u1, c1])
    c4 = layers.Conv2D(16, 3, activation='relu', padding='same')(concat1)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === MAIN TRAINING ===
def main():
    print("[INFO] Generating synthetic dataset...")
    X, Y = [], []
    for _ in range(NUM_SAMPLES):
        label = np.random.rand() < 0.5
        img, mask = generate_synthetic_shell(INPUT_SIZE[0], with_shell=label)
        X.append(img)
        Y.append(mask)

    X = np.array(X)
    Y = np.array(Y)

    print("[INFO] Building U-Net model...")
    model = build_unet(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1))

    print("[INFO] Training...")
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    model.save(OUTPUT_MODEL_PATH)
    print(f"[✅] Model saved to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()