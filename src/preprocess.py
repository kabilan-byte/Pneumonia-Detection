"""
src/preprocess.py — Image Preprocessing
=========================================
Handles loading, resizing, color conversion, and normalization
of chest X-ray images using OpenCV.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import IMG_SIZE, CLASSES, TRAIN_DIR


# ──────────────────────────────────────────────────────────────
# SINGLE IMAGE PREPROCESSING
# ──────────────────────────────────────────────────────────────

def preprocess_image(image_path: str,
                     img_size: tuple = IMG_SIZE,
                     use_rgb: bool = True) -> np.ndarray:
    """
    Load a single chest X-ray image and preprocess it.

    Steps:
        1. Read image from disk with OpenCV
        2. Convert BGR → RGB (or Grayscale)
        3. Resize to target dimensions
        4. Normalize pixel values to [0.0, 1.0]
        5. Add channel dimension if grayscale

    Args:
        image_path (str): Absolute or relative path to the image file.
        img_size   (tuple): Target (width, height). Default: (224, 224).
        use_rgb    (bool): True → RGB (for Transfer Learning),
                           False → Grayscale (for custom CNN).

    Returns:
        np.ndarray: Float32 array of shape (H, W, 3) or (H, W, 1).

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If the image cannot be read (corrupted / unsupported).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image (corrupted or unsupported format): {image_path}")

    # ── Color conversion ──────────────────────────────────────
    if use_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Resize ────────────────────────────────────────────────
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    # ── Normalize ─────────────────────────────────────────────
    img = img.astype(np.float32) / 255.0

    # ── Add channel dim for grayscale ─────────────────────────
    if not use_rgb:
        img = np.expand_dims(img, axis=-1)  # (H, W) → (H, W, 1)

    return img


# ──────────────────────────────────────────────────────────────
# BATCH LOADING
# ──────────────────────────────────────────────────────────────

def load_dataset(directory: str,
                 img_size: tuple = IMG_SIZE,
                 use_rgb: bool = True,
                 class_names: list = None) -> tuple:
    """
    Load all images from a directory with class subfolders.

    Expected structure:
        directory/
            NORMAL/
                img1.jpeg
                img2.jpeg
            PNEUMONIA/
                img3.jpeg
                ...

    Args:
        directory   (str): Root directory containing class subfolders.
        img_size    (tuple): Target image dimensions.
        use_rgb     (bool): RGB or Grayscale.
        class_names (list): Override class names. Defaults to folder names.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            X — float32 image array, shape (N, H, W, C)
            y — int binary label array, shape (N,)
    """
    if class_names is None:
        class_names = sorted(os.listdir(directory))
        class_names = [c for c in class_names if not c.startswith(".")]

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    images, labels = [], []
    total = 0

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue

        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"  Loading '{class_name}': {len(files)} images...", end="", flush=True)

        loaded = 0
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                img = preprocess_image(fpath, img_size=img_size, use_rgb=use_rgb)
                images.append(img)
                labels.append(class_to_idx[class_name])
                loaded += 1
            except (FileNotFoundError, ValueError):
                pass  # Skip bad files silently

        print(f" ✓ ({loaded} loaded)")
        total += loaded

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    print(f"\n✅ Dataset loaded: {total} images | Shape: {X.shape}")
    return X, y


# ──────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────

def show_sample_images(directory: str = TRAIN_DIR,
                       n_per_class: int = 4,
                       img_size: tuple = IMG_SIZE,
                       save_path: str = None):
    """
    Display a grid of sample images from each class.

    Args:
        directory   (str): Dataset directory.
        n_per_class (int): Number of samples to show per class.
        img_size    (tuple): Target size for display.
        save_path   (str): If provided, saves the figure to this path.
    """
    class_names = sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and not d.startswith(".")
    ])

    fig, axes = plt.subplots(len(class_names), n_per_class,
                             figsize=(n_per_class * 3, len(class_names) * 3))
    fig.suptitle("Sample Chest X-Ray Images", fontsize=16, fontweight="bold")

    for row, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))][:n_per_class]

        for col, fname in enumerate(files):
            img = cv2.imread(os.path.join(class_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)

            ax = axes[row][col] if len(class_names) > 1 else axes[col]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(class_name, fontsize=10, color="green" if class_name == "NORMAL" else "red")
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📸 Sample grid saved to: {save_path}")
    plt.show()


def apply_clahe(image_path: str, img_size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    X-ray contrast — often improves model performance on low-contrast scans.

    Args:
        image_path (str): Path to the image.
        img_size   (tuple): Target size.

    Returns:
        np.ndarray: Float32 enhanced image of shape (H, W, 3).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Stack to 3 channels so it matches RGB models
    enhanced_rgb = cv2.merge([enhanced, enhanced, enhanced])
    return enhanced_rgb.astype(np.float32) / 255.0


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    print("=" * 55)
    print("   preprocess.py — Quick Sanity Check")
    print("=" * 55)

    # Find any test image
    sample_paths = glob.glob(os.path.join(TRAIN_DIR, "**", "*.jpeg"), recursive=True)
    if not sample_paths:
        sample_paths = glob.glob(os.path.join(TRAIN_DIR, "**", "*.jpg"), recursive=True)

    if sample_paths:
        sample = sample_paths[0]
        print(f"\n📂 Testing with: {sample}")

        img_rgb  = preprocess_image(sample, use_rgb=True)
        img_gray = preprocess_image(sample, use_rgb=False)

        print(f"   RGB   shape : {img_rgb.shape}  | dtype: {img_rgb.dtype} | range: [{img_rgb.min():.2f}, {img_rgb.max():.2f}]")
        print(f"   Gray  shape : {img_gray.shape} | dtype: {img_gray.dtype} | range: [{img_gray.min():.2f}, {img_gray.max():.2f}]")
        print("\n✅ Preprocessing looks correct!")
    else:
        print("\n⚠️  No images found in TRAIN_DIR. Please add data to data/train/")
