"""
src/model_cnn.py — Custom CNN Architecture
============================================
A hand-crafted Convolutional Neural Network for binary chest X-ray
classification. Three convolutional blocks followed by a dense head.
Good for learning; works well with limited compute.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import INPUT_SHAPE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, Input
)
from tensorflow.keras.regularizers import l2


def build_custom_cnn(input_shape: tuple = INPUT_SHAPE,
                     dropout_rate: float = 0.5,
                     l2_lambda: float = 1e-4) -> Sequential:
    """
    Build a custom CNN for binary pneumonia classification.

    Architecture:
        Block 1 → Conv(32) + BN + MaxPool           (edge / texture detection)
        Block 2 → Conv(64) + BN + MaxPool           (shape / structure detection)
        Block 3 → Conv(128) + BN + MaxPool          (lung-level pattern detection)
        Block 4 → Conv(256) + BN + MaxPool          (high-level features)
        Head    → Flatten → Dense(512) → Dropout → Dense(1, sigmoid)

    Args:
        input_shape  (tuple): Shape of one input image, e.g. (224, 224, 1) or (224, 224, 3).
        dropout_rate (float): Fraction of neurons to randomly zero-out (prevents overfitting).
        l2_lambda    (float): L2 regularization strength.

    Returns:
        tf.keras.Sequential: Compiled-ready model.
    """
    model = Sequential(name="Custom_CNN_Pneumonia")

    # ── Block 1 ──────────────────────────────────────────────────
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 2 ──────────────────────────────────────────────────
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # ── Block 3 ──────────────────────────────────────────────────
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # ── Block 4 ──────────────────────────────────────────────────
    model.add(Conv2D(256, (3, 3), activation="relu", padding="same",
                     kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # ── Classification Head ──────────────────────────────────────
    model.add(Flatten())
    model.add(Dense(512, activation="relu", kernel_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))   # Binary output: P(PNEUMONIA)

    return model


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("=" * 55)
    print("   model_cnn.py — Architecture Summary")
    print("=" * 55)

    model = build_custom_cnn()
    model.summary()

    # Dummy forward pass
    dummy = np.zeros((1,) + INPUT_SHAPE, dtype=np.float32)
    out = model.predict(dummy, verbose=0)
    print(f"\n✅ Forward pass OK | Output shape: {out.shape} | Value: {out[0][0]:.4f}")
    print("   (Value is raw sigmoid probability — not yet trained)")
