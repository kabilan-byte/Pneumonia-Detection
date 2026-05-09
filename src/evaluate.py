"""
src/evaluate.py — Model Evaluation Script
==========================================
Evaluates the trained model on the held-out test set and generates:
    • Classification report (precision / recall / F1 per class)
    • Confusion matrix heatmap   → results/confusion_matrix.png
    • ROC curve + AUC score      → results/roc_curve.png
    • Detailed metrics JSON      → results/metrics.json

Usage:
    python src/evaluate.py                         # Uses config defaults
    python src/evaluate.py --model models/best_model.h5
    python src/evaluate.py --model models/final_model.h5 --batch-size 16
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_SAVE_PATH, FINAL_MODEL_PATH, RESULTS_DIR,
    TEST_DIR, BATCH_SIZE, THRESHOLD, CLASSES
)

import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, accuracy_score, f1_score,
    precision_score, recall_score
)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _load_model(model_path: str) -> tf.keras.Model:
    """Load a Keras model from disk (h5 or SavedModel)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"\n🔄 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully.")
    return model


def _get_test_generator(batch_size: int = BATCH_SIZE):
    """Return a test ImageDataGenerator (no augmentation)."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from config import IMG_SIZE

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    print(f"🟡 Test generator | Classes: {gen.class_indices} | Samples: {gen.samples}")
    return gen


# ──────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: list,
                          save_path: str):
    """Generate and save a nicely styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Confusion Matrix — Pneumonia Detection", fontsize=15, fontweight="bold")

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
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved → {save_path}")


# ──────────────────────────────────────────────────────────────
# ROC CURVE
# ──────────────────────────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray,
                   y_scores: np.ndarray,
                   save_path: str) -> float:
    """Plot ROC curve and return AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#3a86ff", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#e63946", lw=1.5,
            linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#3a86ff")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Pneumonia Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 ROC curve saved → {save_path}  (AUC = {roc_auc:.4f})")
    return roc_auc


# ──────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ──────────────────────────────────────────────────────────────

def evaluate_model(model_path: str = None,
                   batch_size: int = BATCH_SIZE,
                   threshold: float = THRESHOLD):
    """
    Run full evaluation on the test set.

    Args:
        model_path (str): Path to the .h5 model file.
        batch_size (int): Batch size for inference.
        threshold  (float): Decision boundary (>= → PNEUMONIA).

    Returns:
        dict: Dictionary with all computed metrics.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Resolve model path ────────────────────────────────────
    if model_path is None:
        # Prefer best_model over final_model
        model_path = MODEL_SAVE_PATH if os.path.exists(MODEL_SAVE_PATH) else FINAL_MODEL_PATH

    model = _load_model(model_path)

    # ── Get predictions ───────────────────────────────────────
    print("\n🔍 Running inference on test set...")
    test_gen = _get_test_generator(batch_size=batch_size)

    y_scores = model.predict(test_gen, verbose=1).flatten()
    y_true   = test_gen.classes
    y_pred   = (y_scores >= threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────
    acc       = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print("  📋 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # ── Plots ─────────────────────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred, CLASSES,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )
    roc_auc = plot_roc_curve(
        y_true, y_scores,
        save_path=os.path.join(RESULTS_DIR, "roc_curve.png"),
    )

    # ── Save JSON metrics ─────────────────────────────────────
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

    json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Metrics saved → {json_path}")

    return metrics


# ──────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Pneumonia Detection model")
    parser.add_argument("--model",      type=str,   default=None,
                        help="Path to model file (.h5). Defaults to models/best_model.h5")
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE,
                        help="Batch size for inference")
    parser.add_argument("--threshold",  type=float, default=THRESHOLD,
                        help="Decision threshold (default 0.5)")
    args = parser.parse_args()

    metrics = evaluate_model(
        model_path=args.model,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    print("\n✅ Evaluation complete. Results saved to results/")
