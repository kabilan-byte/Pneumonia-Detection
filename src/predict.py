"""
src/predict.py — Single-Image Prediction Script
=================================================
Loads the trained model and runs inference on a single chest X-ray image.
Prints the predicted class and confidence score, and optionally displays
a visual overlay (with Grad-CAM heatmap).

Usage:
    python src/predict.py --image path/to/xray.jpg
    python src/predict.py --image path/to/xray.jpg --model models/best_model.h5
    python src/predict.py --image path/to/xray.jpg --no-gradcam
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_SAVE_PATH, FINAL_MODEL_PATH,
    IMG_SIZE, THRESHOLD, CLASSES, RESULTS_DIR
)

import tensorflow as tf


# ──────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────

def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for model inference.

    Args:
        image_path (str): Path to a .jpg / .jpeg / .png chest X-ray.

    Returns:
        np.ndarray: Float32 array of shape (1, H, W, 3) — batch of 1.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)   # (1, H, W, 3)


# ──────────────────────────────────────────────────────────────
# GRAD-CAM
# ──────────────────────────────────────────────────────────────

def compute_gradcam(model: tf.keras.Model,
                    img_array: np.ndarray,
                    last_conv_layer_name: str = None) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for the given image.

    Returns:
        np.ndarray: Heatmap of shape (H, W) normalized to [0, 1].
                    Returns None if layer cannot be found.
    """
    # Auto-detect last conv layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
            # For transfer learning models with sub-models
            if hasattr(layer, "layers"):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        last_conv_layer_name = sub_layer.name
                        break
                if last_conv_layer_name:
                    break

    if last_conv_layer_name is None:
        return None

    try:
        # Build gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= (np.max(heatmap) + 1e-8)
        return heatmap

    except Exception as e:
        print(f"⚠️  Grad-CAM failed ({e}). Skipping heatmap.")
        return None


def overlay_gradcam(image_path: str,
                    heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Superimpose the Grad-CAM heatmap onto the original image.

    Returns:
        np.ndarray: RGB image with heatmap overlay.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = (img * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return overlay


# ──────────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────────

def predict(image_path: str,
            model_path: str = None,
            threshold: float = THRESHOLD,
            use_gradcam: bool = True,
            save_result: bool = True) -> dict:
    """
    Run inference on a single chest X-ray image.

    Args:
        image_path   (str):   Path to the X-ray image.
        model_path   (str):   Path to the .h5 model file.
        threshold    (float): Decision boundary.
        use_gradcam  (bool):  Generate Grad-CAM heatmap overlay.
        save_result  (bool):  Save result image to results/.

    Returns:
        dict: {
            "image_path": ...,
            "label":      "NORMAL" or "PNEUMONIA",
            "confidence": float  (0.0 – 1.0),
            "score":      raw sigmoid output
        }
    """
    # ── Resolve model path ────────────────────────────────────
    if model_path is None:
        model_path = MODEL_SAVE_PATH if os.path.exists(MODEL_SAVE_PATH) else FINAL_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'.\n"
            "Run  python src/train.py  first."
        )

    # ── Load model + preprocess ───────────────────────────────
    print(f"\n🔄 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"🖼️  Preprocessing: {image_path}")
    img_array = load_and_preprocess(image_path)

    # ── Inference ─────────────────────────────────────────────
    score  = float(model.predict(img_array, verbose=0)[0][0])
    label  = CLASSES[int(score >= threshold)]         # "NORMAL" or "PNEUMONIA"
    conf   = score if label == "PNEUMONIA" else 1 - score   # confidence in prediction

    print("\n" + "=" * 55)
    print("  🫁 PREDICTION RESULT")
    print("=" * 55)
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : {'🔴 ' if label == 'PNEUMONIA' else '🟢 '}{label}")
    print(f"  Confidence : {conf:.2%}")
    print(f"  Raw Score  : {score:.6f}  (threshold = {threshold})")
    print("=" * 55)

    # ── Grad-CAM + save ──────────────────────────────────────
    if use_gradcam or save_result:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        heatmap = compute_gradcam(model, img_array) if use_gradcam else None

        fig, axes = plt.subplots(1, 2 if heatmap is not None else 1, figsize=(12, 5))
        if heatmap is None:
            axes = [axes]

        # Original image
        orig = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, IMG_SIZE)
        axes[0].imshow(orig)
        axes[0].set_title("Input X-Ray", fontsize=12)
        axes[0].axis("off")

        # Grad-CAM overlay
        if heatmap is not None:
            overlay = overlay_gradcam(image_path, heatmap)
            axes[1].imshow(overlay)
            axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
            axes[1].axis("off")

        color   = "#e63946" if label == "PNEUMONIA" else "#2dc653"
        fig.suptitle(
            f"Prediction: {label}  |  Confidence: {conf:.2%}",
            fontsize=14, fontweight="bold", color=color,
        )
        plt.tight_layout()

        if save_result:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(RESULTS_DIR, f"prediction_{basename}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"\n💾 Result image saved → {out_path}")

        plt.close()

    return {
        "image_path": image_path,
        "label":      label,
        "confidence": round(conf, 6),
        "score":      round(score, 6),
    }


# ──────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-image pneumonia prediction with optional Grad-CAM"
    )
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to the chest X-ray image (.jpg / .jpeg / .png)")
    parser.add_argument("--model",      type=str, default=None,
                        help="Path to the .h5 model. Defaults to models/best_model.h5")
    parser.add_argument("--threshold",  type=float, default=THRESHOLD,
                        help="Decision threshold (default 0.5)")
    parser.add_argument("--no-gradcam", action="store_false", dest="use_gradcam",
                        help="Disable Grad-CAM heatmap")
    parser.add_argument("--no-save",    action="store_false", dest="save_result",
                        help="Do not save result image to results/")

    args = parser.parse_args()

    result = predict(
        image_path=args.image,
        model_path=args.model,
        threshold=args.threshold,
        use_gradcam=args.use_gradcam,
        save_result=args.save_result,
    )
