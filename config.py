"""
config.py — Central Configuration
==================================
All project-wide settings live here.
Change values in ONE place instead of hunting through every file.
"""

import os

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
TRAIN_DIR       = os.path.join(DATA_DIR, "train")
VAL_DIR         = os.path.join(DATA_DIR, "val")
TEST_DIR        = os.path.join(DATA_DIR, "test")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

# Auto-create required directories
for _dir in [MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ──────────────────────────────────────────────
# MODEL SETTINGS
# ──────────────────────────────────────────────
MODEL_SAVE_PATH       = os.path.join(MODELS_DIR, "best_model.h5")
FINAL_MODEL_PATH      = os.path.join(MODELS_DIR, "final_model.h5")

# Choose model type: "cnn" | "mobilenetv2" | "resnet50"
MODEL_TYPE = "mobilenetv2"

# ──────────────────────────────────────────────
# IMAGE SETTINGS
# ──────────────────────────────────────────────
IMG_SIZE    = (224, 224)        # (width, height) — standard for MobileNetV2 / ResNet
IMG_CHANNELS = 3                # 3 = RGB (Transfer Learning), 1 = Grayscale (Custom CNN)
INPUT_SHAPE = IMG_SIZE + (IMG_CHANNELS,)   # e.g. (224, 224, 3)

# ──────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ──────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 30
LEARNING_RATE   = 1e-4
FINE_TUNE_LR    = 1e-5          # Lower LR used during fine-tuning phase

# Early stopping / LR scheduler patience
EARLY_STOP_PATIENCE = 7
LR_REDUCE_PATIENCE  = 4
LR_REDUCE_FACTOR    = 0.5
MIN_LR              = 1e-8

# ──────────────────────────────────────────────
# AUGMENTATION
# ──────────────────────────────────────────────
ROTATION_RANGE      = 15
WIDTH_SHIFT         = 0.1
HEIGHT_SHIFT        = 0.1
SHEAR_RANGE         = 0.1
ZOOM_RANGE          = 0.1
HORIZONTAL_FLIP     = True

# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
THRESHOLD   = 0.5               # Decision boundary: >= THRESHOLD → PNEUMONIA
CLASSES     = ["NORMAL", "PNEUMONIA"]
