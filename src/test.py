"""
src/test.py — Model Evaluation / Testing Script
==================================================
இந்த script trained model-ஐ test dataset-ல் evaluate செய்யும்.
(This script evaluates the trained model on the test dataset.)

What it generates (என்ன outputs கொடுக்கும்):
    ✅ Accuracy, Precision, Recall, F1-Score
    📊 Confusion Matrix          → results/confusion_matrix.png
    📈 ROC Curve                 → results/roc_curve.png
    📋 Classification Report     → Terminal-ல் print ஆகும்
    💾 metrics.json              → results/metrics.json
    📸 Training curves summary   → results/test_summary.png

Usage (Terminal-ல்):
    python src/test.py
    python src/test.py --model models/best_model.h5
    python src/test.py --model models/final_model.h5 --batch-size 16
    python src/test.py --threshold 0.4
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Imports
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import argparse
import numpy as np

# Matplotlib — plotting library
import matplotlib
matplotlib.use("Agg")   # Non-interactive (headless) backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Seaborn — prettier plots (confusion matrix heatmap)
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Project path setup
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    MODEL_SAVE_PATH,
    FINAL_MODEL_PATH,
    RESULTS_DIR,
    TEST_DIR,
    BATCH_SIZE,
    THRESHOLD,
    CLASSES,
    IMG_SIZE,
    MODELS_DIR,
)
from src.utils import print_banner, get_best_model_path, Timer

# scikit-learn — metrics computation
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Helper — Build test data generator
#
# Test set-க்கு augmentation வேண்டாம், only normalize பண்ண வேண்டும்.
# (No augmentation for test set — only pixel normalization needed.)
# ─────────────────────────────────────────────────────────────────────────────

def get_test_generator(test_dir: str = TEST_DIR,
                       img_size: tuple = IMG_SIZE,
                       batch_size: int = BATCH_SIZE):
    """
    Create a test data generator (no augmentation, only rescaling).

    Args:
        test_dir   (str): Path to test/ directory.
        img_size   (tuple): Target (width, height).
        batch_size (int): Batch size.

    Returns:
        DirectoryIterator

    Raises:
        FileNotFoundError: If test directory doesn't exist.
    """
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"❌ Test directory not found: '{test_dir}'\n"
            f"   Please ensure data/test/NORMAL/ and data/test/PNEUMONIA/ exist.\n"
            f"   Run:  python download_data.py  to download the dataset."
        )

    # rescale=1/255 → pixel values 0-255 → 0.0-1.0
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    generator = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",   # 0 = NORMAL, 1 = PNEUMONIA
        shuffle=False,          # Keep order for correct label matching
    )

    print(f"🟡 Test generator ready:")
    print(f"   Classes     : {generator.class_indices}")
    print(f"   Total images: {generator.samples}")
    print(f"   Batch size  : {batch_size}")
    print(f"   Steps       : {len(generator)}")

    return generator


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Confusion Matrix Plot
#
# Confusion matrix என்பது model எத்தனை correct/incorrect predict பண்ணியது
# என்பதை visual-ஆக காட்டும்.
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list,
                          save_path: str):
    """
    Plot and save a dual confusion matrix (counts + normalized).

    Args:
        y_true      : True labels (0 = NORMAL, 1 = PNEUMONIA).
        y_pred      : Predicted labels.
        class_names : ["NORMAL", "PNEUMONIA"].
        save_path   : File path to save the PNG.
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Confusion Matrix — Pneumonia Detection",
        fontsize=15, fontweight="bold"
    )

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalised (per class)"],
        ["d", ".2%"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt,
            cmap="Blues", ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.75},
        )
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: ROC Curve Plot
#
# ROC Curve — model எவ்வளவு நன்றாக classes separate பண்ண முடிகிறது
# என்பதை காட்டும். AUC = 1.0 → perfect, AUC = 0.5 → random.
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray,
                   y_scores: np.ndarray,
                   save_path: str) -> float:
    """
    Plot ROC curve and compute AUC score.

    Args:
        y_true   : True binary labels.
        y_scores : Sigmoid scores from model.predict().
        save_path: File path to save the PNG.

    Returns:
        float: AUC score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))

    # ROC curve
    ax.plot(fpr, tpr, color="#3a86ff", lw=2.5,
            label=f"ROC Curve  (AUC = {roc_auc:.4f})")

    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.10, color="#3a86ff")

    # Random baseline
    ax.plot([0, 1], [0, 1], color="#e63946", lw=1.5, linestyle="--",
            label="Random Classifier (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curve — Pneumonia Detection",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📈 ROC curve saved → {save_path}  (AUC = {roc_auc:.4f})")
    return roc_auc


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Metrics bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_bar(metrics: dict, save_path: str):
    """
    Plot a horizontal bar chart of all key metrics.

    Args:
        metrics   (dict): { "accuracy": 0.92, "precision": 0.90, ... }
        save_path (str):  File path to save the PNG.
    """
    keys   = ["accuracy", "precision", "recall", "f1_score", "auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    values = [metrics.get(k, 0.0) for k in keys]
    colors = ["#3a86ff", "#06d6a0", "#ffbe0b", "#fb5607", "#8338ec"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.55)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2%}", va="center", ha="left", fontsize=11, fontweight="bold",
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
    ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.4, lw=1)
    ax.text(0.905, -0.6, "90%", color="gray", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Metrics chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_path: str = None,
    batch_size: int = BATCH_SIZE,
    threshold: float = THRESHOLD,
) -> dict:
    """
    Full evaluation pipeline on the test dataset.

    Steps:
        1) Load the trained model
        2) Run inference on all test images
        3) Compute metrics (accuracy, precision, recall, F1, AUC)
        4) Generate confusion matrix and ROC curve plots
        5) Save metrics to results/metrics.json
        6) Print classification report

    Args:
        model_path (str):   Path to .h5 model. Auto-detects if None.
        batch_size (int):   Batch size for inference.
        threshold  (float): Decision boundary.

    Returns:
        dict: All computed metrics.
    """
    print_banner("Pneumonia Detection — Evaluation")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Resolve model path ─────────────────────────────────────────────
    if model_path is None:
        model_path = get_best_model_path(MODELS_DIR)

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            "❌ No trained model found!\n"
            "   Run:  python src/train.py  first.\n"
            f"   Then check:  {MODELS_DIR}/"
        )

    # ── 2. Load model ─────────────────────────────────────────────────────
    print(f"🔄 Loading model: {os.path.basename(model_path)} ...")
    with Timer("Model loading"):
        model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded.\n")

    # ── 3. Get test generator ─────────────────────────────────────────────
    test_gen = get_test_generator(batch_size=batch_size)

    # ── 4. Inference ──────────────────────────────────────────────────────
    print("\n🔍 Running inference on test set...")
    with Timer("Inference"):
        y_scores = model.predict(test_gen, verbose=1).flatten()

    y_true = test_gen.classes              # Ground-truth labels
    y_pred = (y_scores >= threshold).astype(int)  # Predicted labels

    # ── 5. Compute metrics ────────────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    # ── 6. Print results ──────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  📋  EVALUATION RESULTS")
    print("═" * 62)
    print(f"  Model       : {os.path.basename(model_path)}")
    print(f"  Test samples: {len(y_true)}")
    print(f"  Threshold   : {threshold}")
    print("─" * 62)
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision   : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall      : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1-Score    : {f1:.4f}  ({f1*100:.2f}%)")
    print("═" * 62)
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # ── 7. Plots ──────────────────────────────────────────────────────────
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, CLASSES,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )

    # ROC curve
    roc_auc = plot_roc_curve(
        y_true, y_scores,
        save_path=os.path.join(RESULTS_DIR, "roc_curve.png"),
    )

    # Metrics bar chart
    metrics = {
        "model_path":   model_path,
        "threshold":    threshold,
        "test_samples": int(len(y_true)),
        "accuracy":     round(float(acc), 6),
        "precision":    round(float(precision), 6),
        "recall":       round(float(recall), 6),
        "f1_score":     round(float(f1), 6),
        "auc":          round(float(roc_auc), 6),
    }

    plot_metrics_bar(
        metrics,
        save_path=os.path.join(RESULTS_DIR, "metrics_chart.png"),
    )

    # ── 8. Save metrics JSON ──────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Metrics saved → {json_path}")

    # ── 9. Summary ────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  ✅  Evaluation complete!")
    print("─" * 62)
    print(f"  Outputs saved to:  {RESULTS_DIR}/")
    print(f"    • confusion_matrix.png")
    print(f"    • roc_curve.png")
    print(f"    • metrics_chart.png")
    print(f"    • metrics.json")
    print("═" * 62 + "\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🫁 Pneumonia Detection — Model Evaluation on Test Set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/test.py
  python src/test.py --model models/best_model.h5
  python src/test.py --model models/final_model.h5 --threshold 0.45
  python src/test.py --batch-size 16
        """,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        metavar="MODEL_PATH",
        help="Path to the .h5 model file. Auto-detects best model if not set.",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=BATCH_SIZE,
        metavar="N",
        help=f"Batch size for inference (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=THRESHOLD,
        metavar="THRESHOLD",
        help=f"Decision threshold (default: {THRESHOLD})",
    )

    args = parser.parse_args()

    try:
        results = evaluate_model(
            model_path=args.model,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
    except FileNotFoundError as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
