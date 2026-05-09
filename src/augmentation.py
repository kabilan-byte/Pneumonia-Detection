"""
src/augmentation.py — Data Generators with Augmentation
=========================================================
Wraps Keras ImageDataGenerator to supply augmented training batches
and clean validation/test batches.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE,
                    ROTATION_RANGE, WIDTH_SHIFT, HEIGHT_SHIFT,
                    SHEAR_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP)

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ──────────────────────────────────────────────────────────────
# TRAINING GENERATOR  (with augmentation)
# ──────────────────────────────────────────────────────────────

def get_train_generator(train_dir: str = TRAIN_DIR,
                        img_size: tuple = IMG_SIZE,
                        batch_size: int = BATCH_SIZE):
    """
    Create an augmented data generator for the training set.

    Augmentations applied:
        - Random rotation  ±15°
        - Horizontal/vertical shift  10%
        - Shear & zoom  10%
        - Horizontal flip (anatomically valid for chest X-rays)
        - Pixel normalization  [0, 1]

    Args:
        train_dir  (str): Path to training data root.
        img_size   (tuple): Target (width, height).
        batch_size (int): Number of images per batch.

    Returns:
        DirectoryIterator: Yields (batch_images, batch_labels).
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT,
        height_shift_range=HEIGHT_SHIFT,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        fill_mode="nearest",            # Fill empty pixels after rotation/shift
        brightness_range=[0.85, 1.15],  # Simulate different exposure levels
    )

    generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",    # 0 = NORMAL, 1 = PNEUMONIA
        shuffle=True,
        seed=42,
    )

    print(f"🟢 Train generator  | Classes: {generator.class_indices} | "
          f"Samples: {generator.samples} | Batches/epoch: {len(generator)}")
    return generator


# ──────────────────────────────────────────────────────────────
# VALIDATION GENERATOR  (no augmentation)
# ──────────────────────────────────────────────────────────────

def get_val_generator(val_dir: str = VAL_DIR,
                      img_size: tuple = IMG_SIZE,
                      batch_size: int = BATCH_SIZE):
    """
    Create a clean (no augmentation) data generator for the validation set.

    Only rescaling is applied — augmentation would introduce noise into
    the validation signal we rely on for early stopping and LR scheduling.

    Args:
        val_dir    (str): Path to validation data root.
        img_size   (tuple): Target (width, height).
        batch_size (int): Number of images per batch.

    Returns:
        DirectoryIterator: Yields (batch_images, batch_labels).
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    generator = datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,          # Keep order for evaluation reproducibility
    )

    print(f"🔵 Val generator    | Classes: {generator.class_indices} | "
          f"Samples: {generator.samples} | Batches/epoch: {len(generator)}")
    return generator


# ──────────────────────────────────────────────────────────────
# TEST GENERATOR  (no augmentation)
# ──────────────────────────────────────────────────────────────

def get_test_generator(test_dir: str = TEST_DIR,
                       img_size: tuple = IMG_SIZE,
                       batch_size: int = BATCH_SIZE):
    """
    Create a clean data generator for the test (holdout) set.

    Args:
        test_dir   (str): Path to test data root.
        img_size   (tuple): Target (width, height).
        batch_size (int): Number of images per batch.

    Returns:
        DirectoryIterator: Yields (batch_images, batch_labels).
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    generator = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    print(f"🟡 Test generator   | Classes: {generator.class_indices} | "
          f"Samples: {generator.samples} | Batches/epoch: {len(generator)}")
    return generator


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("=" * 55)
    print("   augmentation.py — Quick Sanity Check")
    print("=" * 55)

    try:
        train_gen = get_train_generator(batch_size=8)
        batch_x, batch_y = next(train_gen)
        print(f"\n✅ Batch shape : {batch_x.shape}")
        print(f"   Label shape : {batch_y.shape}")
        print(f"   Pixel range : [{batch_x.min():.3f}, {batch_x.max():.3f}]")
        print(f"   Labels      : {batch_y.astype(int).tolist()}")
    except Exception as e:
        print(f"\n⚠️  Could not create generator: {e}")
        print("   Make sure data/ directory exists with NORMAL/ and PNEUMONIA/ subfolders.")
