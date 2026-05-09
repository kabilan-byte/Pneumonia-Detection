"""
src/utils.py — Shared Utility Functions
=========================================
இந்த file-ல் project எங்கும் உபயோகிக்கப்படும் helper functions உள்ளன.
(This file contains helper functions used across the entire project.)

Functions:
    • print_banner()         — Styled title banner
    • check_dataset()        — Validate dataset folder structure
    • count_images()         — Count images per class
    • save_plot()            — Save matplotlib figure to disk
    • set_seed()             — Fix random seeds for reproducibility
    • format_time()          — Human-readable elapsed time
    • get_class_weights()    — Compute class weights for imbalanced data
    • list_model_files()     — List available trained models
"""

import os
import sys
import time
import random
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(title: str = "Pneumonia Detection AI"):
    """
    Print a styled ASCII banner.

    Usage:
        print_banner("Training Started")
    """
    width = 62
    border = "═" * width
    print(f"\n╔{border}╗")
    print(f"║{'🫁  ' + title:^{width}}║")
    print(f"╚{border}╝\n")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def count_images(directory: str) -> dict:
    """
    Count images inside each class subfolder.

    Args:
        directory (str): Root path containing NORMAL/ and PNEUMONIA/ folders.

    Returns:
        dict: { "NORMAL": 1341, "PNEUMONIA": 3875, ... }

    Example:
        counts = count_images("data/train")
        # → { "NORMAL": 1341, "PNEUMONIA": 3875 }
    """
    result = {}
    if not os.path.isdir(directory):
        return result

    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path) or class_name.startswith("."):
            continue

        count = sum(
            1 for f in os.listdir(class_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        )
        result[class_name] = count

    return result


def check_dataset(data_dir: str, splits: list = None) -> bool:
    """
    Validate the dataset folder structure before training.

    இந்த function dataset folder சரியாக இருக்கிறதா என்று check செய்யும்.
    (This function checks if the dataset folder is properly structured.)

    Expected structure:
        data_dir/
            train/
                NORMAL/     ← images here
                PNEUMONIA/  ← images here
            val/
                NORMAL/
                PNEUMONIA/
            test/
                NORMAL/
                PNEUMONIA/

    Args:
        data_dir (str): Root data directory (e.g. "data/" or "dataset/").
        splits   (list): Subfolders to check. Defaults to ["train", "val", "test"].

    Returns:
        bool: True if all folders exist and contain images, False otherwise.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    required_classes = ["NORMAL", "PNEUMONIA"]
    all_ok = True

    print("📂 Dataset Structure Check")
    print("─" * 50)

    if not os.path.isdir(data_dir):
        print(f"  ❌ Data directory not found: '{data_dir}'")
        print(f"\n  💡 SOLUTION (Tamil / English):")
        print(f"     Dataset download பண்ணு:")
        print(f"     Run:  python download_data.py")
        return False

    for split in splits:
        split_path = os.path.join(data_dir, split)

        if not os.path.isdir(split_path):
            print(f"  ⚠️  Missing split folder: {split_path}")
            all_ok = False
            continue

        counts = count_images(split_path)
        if not counts:
            print(f"  ⚠️  No class folders found in: {split_path}")
            all_ok = False
            continue

        for cls in required_classes:
            n = counts.get(cls, 0)
            icon = "✅" if n > 0 else "❌"
            print(f"  {icon}  {split:6s} / {cls:<10s} → {n:>5d} images")
            if n == 0:
                all_ok = False

    print("─" * 50)
    if all_ok:
        print("  ✅ Dataset structure is valid. Ready to train!\n")
    else:
        print("  ❌ Dataset issues found. Please check the paths above.\n")
        print("  💡 Run:  python download_data.py  to download the dataset.\n")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """
    Fix all random seeds for reproducibility.

    இதை call பண்ணினால் ஒவ்வொரு run-லயும் same results வரும்.
    (Calling this ensures consistent results across every run.)

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"🎲 Random seed fixed to {seed} for reproducibility.")


# ─────────────────────────────────────────────────────────────────────────────
# TIMING
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """
    Simple context-manager / manual timer.

    Usage (automatic):
        with Timer("Training"):
            model.fit(...)
        # → ⏱️  Training completed in 4m 32s

    Usage (manual):
        t = Timer()
        t.start()
        ...
        t.stop("My task")
    """
    def __init__(self, label: str = ""):
        self.label = label
        self._start: float = 0.0

    def start(self):
        self._start = time.time()
        return self

    def stop(self, label: str = None) -> float:
        elapsed = time.time() - self._start
        print(f"⏱️  {label or self.label} completed in {format_time(elapsed)}")
        return elapsed

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        self.stop(self.label)


def format_time(seconds: float) -> str:
    """
    Convert seconds into a human-readable string.

    Args:
        seconds (float): Elapsed time in seconds.

    Returns:
        str: e.g. "3h 12m 05s" or "4m 32s" or "45s"
    """
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m {s % 60:02d}s"
    elif s >= 60:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s}s"


# ─────────────────────────────────────────────────────────────────────────────
# PLOT SAVING
# ─────────────────────────────────────────────────────────────────────────────

def save_plot(fig, filename: str, output_dir: str, dpi: int = 150):
    """
    Save a matplotlib figure to disk, creating directories as needed.

    Args:
        fig        : matplotlib Figure object.
        filename   (str): Output filename (e.g. "training_curves.png").
        output_dir (str): Directory to save into.
        dpi        (int): Image resolution.

    Returns:
        str: Full path where the file was saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"💾 Plot saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS (for imbalanced datasets)
# ─────────────────────────────────────────────────────────────────────────────

def get_class_weights(train_dir: str) -> dict:
    """
    Compute class weights to handle imbalanced datasets.

    Chest X-ray dataset-ல் Pneumonia images அதிகமாக இருக்கும்.
    இந்த function அதை balance பண்ணும்.
    (Pneumonia images are more in the dataset. This balances training.)

    Formula:
        weight_for_class_i = total_samples / (num_classes * samples_in_class_i)

    Args:
        train_dir (str): Training data directory.

    Returns:
        dict: { 0: 1.24, 1: 0.87 }  (index → weight)
    """
    counts = count_images(train_dir)
    if not counts:
        return {}

    class_names = sorted(counts.keys())  # ['NORMAL', 'PNEUMONIA']
    total = sum(counts.values())
    n_classes = len(class_names)

    weights = {}
    for idx, cls in enumerate(class_names):
        weights[idx] = total / (n_classes * counts[cls])

    print("⚖️  Class weights computed:")
    for idx, cls in enumerate(class_names):
        print(f"   Class {idx} ({cls}): {weights[idx]:.4f}")

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FILE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def list_model_files(models_dir: str) -> list:
    """
    List all .h5 / SavedModel files in the models/ directory.

    Args:
        models_dir (str): Path to the models directory.

    Returns:
        list: Sorted list of model file paths.
    """
    if not os.path.isdir(models_dir):
        return []

    model_files = []
    for f in sorted(os.listdir(models_dir)):
        if f.endswith(".h5") or f.endswith(".keras"):
            model_files.append(os.path.join(models_dir, f))

    return model_files


def get_best_model_path(models_dir: str) -> str | None:
    """
    Return the path to the best available trained model.

    Priority: best_model.h5 → final_model.h5 → any .h5 file

    Args:
        models_dir (str): Path to models directory.

    Returns:
        str | None: Path to model file, or None if not found.
    """
    candidates = ["best_model.h5", "final_model.h5", "pneumonia_model.h5"]
    for name in candidates:
        path = os.path.join(models_dir, name)
        if os.path.exists(path):
            return path

    # Fall back to any .h5 in the folder
    files = list_model_files(models_dir)
    return files[0] if files else None


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_banner("utils.py — Self Test")

    # Timer demo
    set_seed(42)

    with Timer("Demo sleep"):
        time.sleep(0.5)

    print(format_time(3750))   # → "1h 02m 30s"
    print(format_time(272))    # → "4m 32s"
    print(format_time(45))     # → "45s"

    # Dataset check
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    try:
        from config import DATA_DIR, MODELS_DIR
        check_dataset(DATA_DIR)
        print("\nAvailable models:")
        for m in list_model_files(MODELS_DIR):
            size_mb = os.path.getsize(m) / 1e6
            print(f"  • {os.path.basename(m)}  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"⚠️  Could not load config: {e}")
