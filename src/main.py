"""
src/main.py — Beginner-Friendly Prediction Script
====================================================
இந்த script ஒரு chest X-ray image-ஐ எடுத்து NORMAL அல்லது PNEUMONIA
என்று classify செய்யும். Beginners-க்காக detailed comments உள்ளன.

(This script takes one chest X-ray image and predicts NORMAL or PNEUMONIA.
 Detailed comments are included for beginners.)

Usage (Terminal-ல் இப்படி run பண்ணுங்க):
    python src/main.py --image path/to/xray.jpg
    python src/main.py --image path/to/xray.jpg --threshold 0.4
    python src/main.py --image path/to/xray.jpg --no-gradcam

Example:
    python src/main.py --image data/test/PNEUMONIA/person1_virus_6.jpeg
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Import libraries (தேவையான libraries import செய்)
# ─────────────────────────────────────────────────────────────────────────────

import os          # File path operations
import sys         # Python path manipulation
import argparse    # Command-line argument parsing
import numpy as np # Numerical operations

# OpenCV — Image loading and preprocessing
# (X-ray image-ஐ load செய்து resize, normalize பண்ண பயன்படும்)
import cv2

# Matplotlib — Visualization (graphs, image display)
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (server-safe)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Add project root to Python path
# (Project root-ஐ Python path-ல் add பண்ண வேண்டும், import work ஆக)
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Import project-specific modules
# ─────────────────────────────────────────────────────────────────────────────

from config import (
    MODEL_SAVE_PATH,    # Best model path: models/best_model.h5
    FINAL_MODEL_PATH,   # Final model path: models/final_model.h5
    IMG_SIZE,           # Input image size: (224, 224)
    THRESHOLD,          # Decision threshold: 0.5
    CLASSES,            # ["NORMAL", "PNEUMONIA"]
    RESULTS_DIR,        # Output directory: results/
    MODELS_DIR,         # Models directory: models/
)

from src.utils import (
    print_banner,
    check_dataset,
    get_best_model_path,
    Timer,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Helper — Load and preprocess image using OpenCV
#
# OpenCV-ஐ பயன்படுத்தி image-ஐ:
#   1) Disk-லிருந்து load செய்யும்
#   2) BGR → RGB convert பண்ணும் (OpenCV BGR format-ல் படிக்கும்)
#   3) 224×224 pixels-க்கு resize பண்ணும்
#   4) Pixel values 0-255 → 0.0-1.0 normalize பண்ணும்
#   5) Model-க்கு batch dimension add பண்ணும்: (H,W,3) → (1,H,W,3)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    Load a chest X-ray image from disk and prepare it for model inference.

    Args:
        image_path (str): Path to the .jpg / .jpeg / .png image.

    Returns:
        np.ndarray: Float32 tensor of shape (1, 224, 224, 3).

    Raises:
        FileNotFoundError: If the image path doesn't exist.
        ValueError: If the image is corrupted or unreadable.
    """
    # ── Check file exists ─────────────────────────────────────────────────
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"❌ Image not found: '{image_path}'\n"
            "   Please provide a valid chest X-ray image path."
        )

    # ── Read image with OpenCV ────────────────────────────────────────────
    # cv2.imread() → BGR format-ல் படிக்கும்
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(
            f"❌ Could not read image: '{image_path}'\n"
            "   File may be corrupted or in an unsupported format."
        )

    # ── BGR → RGB conversion ──────────────────────────────────────────────
    # OpenCV BGR-ல் படிக்கும், ஆனால் model RGB expect பண்ணும்
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Resize to model input size ────────────────────────────────────────
    # IMG_SIZE = (224, 224) — MobileNetV2 / ResNet50 standard input
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # ── Normalize pixel values ────────────────────────────────────────────
    # 0-255 → 0.0-1.0 (model float input expect பண்ணும்)
    img = img.astype(np.float32) / 255.0

    # ── Add batch dimension ───────────────────────────────────────────────
    # (224, 224, 3) → (1, 224, 224, 3)  ← model batch format
    return np.expand_dims(img, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Load the trained model
# (Train பண்ணப்பட்ட model-ஐ disk-லிருந்து load செய்யும்)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str = None):
    """
    Load the trained Keras model from disk.

    Args:
        model_path (str): Optional explicit path. Auto-detects if None.

    Returns:
        tf.keras.Model: Loaded model ready for inference.

    Raises:
        FileNotFoundError: If no trained model is found.
    """
    import tensorflow as tf  # Import here to keep startup fast

    # ── Auto-detect model path ────────────────────────────────────────────
    if model_path is None:
        model_path = get_best_model_path(MODELS_DIR)

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            "❌ No trained model found!\n\n"
            "   முதலில் model train பண்ண வேண்டும்:\n"
            "   (First, train the model:)\n\n"
            "   →  python src/train.py\n\n"
            "   அல்லது (or) download பண்ணிய model வை:\n"
            "   →  models/ folder-ல் .h5 file வையுங்கள்"
        )

    print(f"🔄 Loading model: {os.path.basename(model_path)} ...")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully!")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Run inference (Prediction)
# (Model-ஐ வைத்து image-ஐ classify பண்ணும்)
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, img_array: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """
    Run the model on a preprocessed image and return prediction details.

    How it works (எப்படி வேலை செய்கிறது):
        • Model output: sigmoid score between 0.0 and 1.0
        • score < threshold  → NORMAL    (healthy lung)
        • score >= threshold → PNEUMONIA (infection detected)

    Args:
        model     : Loaded tf.keras.Model.
        img_array : np.ndarray of shape (1, H, W, 3).
        threshold : Decision boundary (default 0.5).

    Returns:
        dict: {
            "label":      "NORMAL" or "PNEUMONIA",
            "confidence": float (0.0 – 1.0),
            "score":      raw sigmoid output
        }
    """
    # ── Get raw sigmoid score ─────────────────────────────────────────────
    # verbose=0 → silent prediction (no progress bar)
    score = float(model.predict(img_array, verbose=0)[0][0])

    # ── Apply threshold ───────────────────────────────────────────────────
    # CLASSES = ["NORMAL", "PNEUMONIA"]
    # int(score >= threshold) → 0 (NORMAL) or 1 (PNEUMONIA)
    label = CLASSES[int(score >= threshold)]

    # ── Confidence = how sure the model is ───────────────────────────────
    confidence = score if label == "PNEUMONIA" else (1.0 - score)

    return {
        "label":      label,
        "confidence": confidence,
        "score":      score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Grad-CAM heatmap (Optional — shows where model is "looking")
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradcam(model, img_array: np.ndarray) -> np.ndarray | None:
    """
    Compute Grad-CAM heatmap — highlights which regions of the X-ray the
    model focused on while making its prediction.

    Grad-CAM என்பது model எந்த lung region-ஐ பார்த்து decision எடுத்தது
    என்பதை visualize பண்ண பயன்படும் technique.

    Returns:
        np.ndarray: Normalized heatmap (H, W) or None on failure.
    """
    import tensorflow as tf

    # Find the last convolutional layer in the model
    # (கடைசி Conv layer-ஐ தேடும் — feature map இங்கே இருக்கும்)
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    last_conv_layer = sub.name
                    break
            if last_conv_layer:
                break

    if last_conv_layer is None:
        return None

    try:
        # Build gradient model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output],
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv outputs by gradients
        heatmap = (conv_outputs[0] @ pooled_grads[..., tf.newaxis]).numpy().squeeze()
        heatmap = np.maximum(heatmap, 0)      # ReLU
        heatmap /= (heatmap.max() + 1e-8)    # Normalize to [0, 1]
        return heatmap

    except Exception as e:
        print(f"⚠️  Grad-CAM failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Display and save the prediction result
# ─────────────────────────────────────────────────────────────────────────────

def display_and_save_result(
    image_path: str,
    result: dict,
    heatmap: np.ndarray = None,
    save: bool = True,
) -> str | None:
    """
    Create a visual summary of the prediction and save it.

    Left panel  : Original X-ray
    Right panel : Grad-CAM overlay (if available)
    Title       : Prediction + confidence

    Args:
        image_path (str): Path to the original image.
        result     (dict): Output from run_inference().
        heatmap    (np.ndarray): Grad-CAM heatmap (optional).
        save       (bool): Save figure to results/ folder.

    Returns:
        str | None: Saved file path, or None if not saved.
    """
    label      = result["label"]
    confidence = result["confidence"]
    color      = "#e63946" if label == "PNEUMONIA" else "#2dc653"
    icon       = "🔴" if label == "PNEUMONIA" else "🟢"

    # ── Load original image for display ──────────────────────────────────
    orig = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, IMG_SIZE)

    n_panels = 2 if heatmap is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: Original ─────────────────────────────────────────────────
    axes[0].imshow(orig)
    axes[0].set_title("Input Chest X-Ray", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # ── Panel 2: Grad-CAM overlay ─────────────────────────────────────────
    if heatmap is not None:
        heat_resized  = cv2.resize(heatmap, IMG_SIZE)
        heat_uint8    = np.uint8(255 * heat_resized)
        heat_colored  = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_colored  = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
        overlay       = (orig * 0.55 + heat_colored * 0.45).astype(np.uint8)
        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM Attention Map", fontsize=12, fontweight="bold")
        axes[1].axis("off")

    # ── Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"{icon}  Prediction: {label}   |   Confidence: {confidence:.1%}",
        fontsize=14, fontweight="bold", color=color, y=1.01,
    )
    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = None
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(RESULTS_DIR, f"prediction_{basename}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n💾 Result image saved → {out_path}")

    plt.close()
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9: Main prediction function (everything brought together)
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(
    image_path: str,
    model_path: str = None,
    threshold: float = THRESHOLD,
    use_gradcam: bool = True,
    save_result: bool = True,
) -> dict:
    """
    Full end-to-end prediction pipeline.

    ஒரு image path கொடுத்தால், இந்த function முழு pipeline run பண்ணும்:
        1) Image load + preprocess
        2) Model load
        3) Inference (prediction)
        4) Grad-CAM heatmap
        5) Result display + save

    Args:
        image_path  (str):   Path to chest X-ray (.jpg/.jpeg/.png).
        model_path  (str):   Optional model path (auto-detects if None).
        threshold   (float): Decision boundary.
        use_gradcam (bool):  Generate Grad-CAM heatmap.
        save_result (bool):  Save result image to results/ folder.

    Returns:
        dict: Prediction result with keys: label, confidence, score.
    """
    print_banner("Pneumonia Detection — Prediction")

    # ── 1. Preprocess image ───────────────────────────────────────────────
    print(f"🖼️   Image    : {image_path}")
    img_array = load_and_preprocess(image_path)
    print(f"📐  Input shape: {img_array.shape}  | dtype: {img_array.dtype}")

    # ── 2. Load model ─────────────────────────────────────────────────────
    with Timer("Model loading"):
        model = load_model(model_path)

    # ── 3. Inference ──────────────────────────────────────────────────────
    with Timer("Inference"):
        result = run_inference(model, img_array, threshold=threshold)

    # ── 4. Print result ───────────────────────────────────────────────────
    label      = result["label"]
    confidence = result["confidence"]
    score      = result["score"]
    icon       = "🔴" if label == "PNEUMONIA" else "🟢"

    print("\n" + "═" * 58)
    print("  🫁  PREDICTION RESULT")
    print("═" * 58)
    print(f"  Image       : {os.path.basename(image_path)}")
    print(f"  Prediction  : {icon}  {label}")
    print(f"  Confidence  : {confidence:.2%}")
    print(f"  Raw Score   : {score:.6f}  (threshold = {threshold})")
    print(f"  Meaning     : {'Pneumonia detected! ⚕️' if label == 'PNEUMONIA' else 'Lungs appear normal ✅'}")
    print("═" * 58)

    if label == "PNEUMONIA":
        print("\n  ⚠️  DISCLAIMER: இது AI prediction மட்டுமே.")
        print("     (This is AI prediction only.)")
        print("     Please consult a qualified doctor for diagnosis.")
    else:
        print("\n  ✅ Normal prediction. Always verify with a doctor.")

    # ── 5. Grad-CAM + save ────────────────────────────────────────────────
    heatmap = None
    if use_gradcam:
        print("\n🔥 Computing Grad-CAM heatmap...")
        heatmap = compute_gradcam(model, img_array)
        if heatmap is None:
            print("   ⚠️  Grad-CAM not available for this model architecture.")

    if use_gradcam or save_result:
        display_and_save_result(image_path, result, heatmap, save=save_result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: CLI entry point
# (Terminal-ல் arguments கொண்டு run பண்ணுவதற்கு)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Argument parser ───────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="🫁 Pneumonia Detection — Single Image Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --image data/test/PNEUMONIA/person1_virus_6.jpeg
  python src/main.py --image xray.jpg --threshold 0.4
  python src/main.py --image xray.jpg --no-gradcam --no-save
        """,
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        metavar="IMAGE_PATH",
        help="Path to the chest X-ray image (.jpg / .jpeg / .png)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        metavar="MODEL_PATH",
        help="Path to the trained model (.h5). Auto-detects if not set.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=THRESHOLD,
        metavar="THRESHOLD",
        help=f"Decision boundary (default: {THRESHOLD}). "
             "Scores ≥ this value → PNEUMONIA",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_false",
        dest="use_gradcam",
        help="Disable Grad-CAM heatmap generation",
    )
    parser.add_argument(
        "--no-save",
        action="store_false",
        dest="save_result",
        help="Do not save result image to results/ folder",
    )

    args = parser.parse_args()

    # ── Run prediction ────────────────────────────────────────────────────
    try:
        result = predict_image(
            image_path=args.image,
            model_path=args.model,
            threshold=args.threshold,
            use_gradcam=args.use_gradcam,
            save_result=args.save_result,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
