"""
src/model_transfer.py — Transfer Learning Models
==================================================
Pre-trained MobileNetV2 and ResNet50 backbones with custom classification
heads for binary pneumonia detection.

Strategy:
    Phase 1 — Train only the custom head (base frozen).
    Phase 2 — Unfreeze last N layers of base and fine-tune with very low LR.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import INPUT_SHAPE

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.regularizers import l2


# ──────────────────────────────────────────────────────────────
# MobileNetV2
# ──────────────────────────────────────────────────────────────

def build_mobilenetv2(input_shape: tuple = INPUT_SHAPE,
                      dropout_rate: float = 0.3) -> Model:
    """
    MobileNetV2 backbone + custom binary classification head.

    Why MobileNetV2?
        - Lightweight (~3.4M params) — fast on CPU/GPU
        - Excellent accuracy for its size
        - Great for mobile / embedded deployment

    Args:
        input_shape  (tuple): e.g. (224, 224, 3)
        dropout_rate (float): Dropout before final Dense layer.

    Returns:
        tf.keras.Model: Transfer learning model (base frozen by default).
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,      # Remove ImageNet's 1000-class head
        weights="imagenet",     # Use pre-trained ImageNet weights
    )
    base.trainable = False      # Freeze all base layers initially

    # ── Custom head ──────────────────────────────────────────────
    inputs  = Input(shape=input_shape)
    x       = base(inputs, training=False)
    x       = GlobalAveragePooling2D()(x)
    x       = Dense(256, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x       = BatchNormalization()(x)
    x       = Dropout(dropout_rate)(x)
    x       = Dense(64, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="MobileNetV2_Pneumonia")
    return model


# ──────────────────────────────────────────────────────────────
# ResNet50
# ──────────────────────────────────────────────────────────────

def build_resnet50(input_shape: tuple = INPUT_SHAPE,
                   dropout_rate: float = 0.4) -> Model:
    """
    ResNet50 backbone + custom binary classification head.

    Why ResNet50?
        - Deep residual connections avoid vanishing gradients
        - ~25M params — more powerful than MobileNetV2
        - Often achieves highest accuracy on medical imaging

    Args:
        input_shape  (tuple): e.g. (224, 224, 3)
        dropout_rate (float): Dropout before final Dense layer.

    Returns:
        tf.keras.Model: Transfer learning model (base frozen by default).
    """
    base = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs  = Input(shape=input_shape)
    x       = base(inputs, training=False)
    x       = GlobalAveragePooling2D()(x)
    x       = Dense(512, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x       = BatchNormalization()(x)
    x       = Dropout(dropout_rate)(x)
    x       = Dense(128, activation="relu")(x)
    x       = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="ResNet50_Pneumonia")
    return model


# ──────────────────────────────────────────────────────────────
# EfficientNetB0  (bonus — often best accuracy/param ratio)
# ──────────────────────────────────────────────────────────────

def build_efficientnetb0(input_shape: tuple = INPUT_SHAPE,
                          dropout_rate: float = 0.3) -> Model:
    """
    EfficientNetB0 backbone + custom binary classification head.

    Why EfficientNetB0?
        - State-of-the-art accuracy with fewer parameters
        - Scales efficiently (B0 → B7)
        - Best accuracy/compute tradeoff

    Args:
        input_shape  (tuple): e.g. (224, 224, 3)
        dropout_rate (float): Dropout before final Dense layer.

    Returns:
        tf.keras.Model: Transfer learning model (base frozen by default).
    """
    base = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs  = Input(shape=input_shape)
    x       = base(inputs, training=False)
    x       = GlobalAveragePooling2D()(x)
    x       = Dense(256, activation="relu")(x)
    x       = BatchNormalization()(x)
    x       = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="EfficientNetB0_Pneumonia")
    return model


# ──────────────────────────────────────────────────────────────
# FINE-TUNING HELPER
# ──────────────────────────────────────────────────────────────

def unfreeze_top_layers(model: Model,
                        num_layers: int = 30,
                        verbose: bool = True) -> Model:
    """
    Unfreeze the last `num_layers` of the base model for fine-tuning.

    Call this AFTER Phase 1 training is complete. Then re-compile with a
    lower learning rate (e.g. 1e-5) before continuing training.

    Args:
        model      (tf.keras.Model): The full transfer learning model.
        num_layers (int): Number of layers from the END of base to unfreeze.
        verbose    (bool): Print which layers are unfrozen.

    Returns:
        tf.keras.Model: Model with some base layers unfrozen.
    """
    # Find the base model (first layer that is itself a model)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers"):    # It's a sub-model
            base_model = layer
            break

    if base_model is None:
        print("⚠️  Could not find base model inside the model.")
        return model

    # Unfreeze last num_layers
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    unfrozen = [l.name for l in base_model.layers if l.trainable]
    if verbose:
        print(f"\n🔓 Fine-tuning: unfroze last {num_layers} layers of {base_model.name}")
        print(f"   Unfrozen layers ({len(unfrozen)}): {unfrozen[-5:]} ... (last 5 shown)")

    return model


# ──────────────────────────────────────────────────────────────
# MODEL FACTORY
# ──────────────────────────────────────────────────────────────

def get_transfer_model(model_type: str = "mobilenetv2",
                       input_shape: tuple = INPUT_SHAPE) -> Model:
    """
    Factory function: return the right model by name string.

    Args:
        model_type  (str): "mobilenetv2" | "resnet50" | "efficientnetb0"
        input_shape (tuple): Input image shape.

    Returns:
        tf.keras.Model
    """
    builders = {
        "mobilenetv2":    build_mobilenetv2,
        "resnet50":       build_resnet50,
        "efficientnetb0": build_efficientnetb0,
    }

    model_type = model_type.lower()
    if model_type not in builders:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(builders.keys())}")

    print(f"\n🧠 Building Transfer Learning model: {model_type.upper()}")
    return builders[model_type](input_shape=input_shape)


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    for name in ["mobilenetv2", "resnet50", "efficientnetb0"]:
        print(f"\n{'='*55}")
        model = get_transfer_model(name)
        dummy = np.zeros((1,) + INPUT_SHAPE, dtype=np.float32)
        out = model.predict(dummy, verbose=0)
        total_params = model.count_params()
        trainable    = sum(tf.keras.backend.count_params(p) for p in model.trainable_weights)
        print(f"   Output: {out.shape} | Value: {out[0][0]:.4f}")
        print(f"   Total params    : {total_params:,}")
        print(f"   Trainable params: {trainable:,}  (base is frozen)")
