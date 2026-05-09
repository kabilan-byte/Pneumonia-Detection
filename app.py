"""
app.py — Streamlit Web UI for Pneumonia Detection
===================================================
A premium, dark-themed web application for chest X-ray classification.

Features:
    • Drag-and-drop image upload
    • Real-time inference with confidence gauge
    • Grad-CAM heatmap overlay
    • Model performance metrics sidebar
    • Batch prediction from a folder

Launch:
    streamlit run app.py
"""

import os
import sys
import json
import tempfile
import numpy as np
import cv2
from PIL import Image
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    MODEL_SAVE_PATH, FINAL_MODEL_PATH, RESULTS_DIR,
    IMG_SIZE, THRESHOLD, CLASSES
)

# ── Must be the FIRST Streamlit call ──────────────────────────
st.set_page_config(
    page_title="🫁 Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS  — premium dark theme
# ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Background ── */
  .stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b27 50%, #0d1117 100%);
    color: #e2e8f0;
  }

  /* ── Main container ── */
  .main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
  }

  /* ── Hero title ── */
  .hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
  }
  .hero-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2.5rem;
  }

  /* ── Upload area ── */
  [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed #334155;
    border-radius: 16px;
    padding: 1.5rem;
    transition: border-color 0.3s;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #60a5fa;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
  }
  .metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }

  /* ── Result badge ── */
  .result-normal {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
  }
  .result-pneumonia {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
  }
  .result-label {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
  }
  .result-conf {
    font-size: 1.1rem;
    opacity: 0.85;
  }

  /* ── Divider ── */
  hr { border-color: rgba(255,255,255,0.08); }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: rgba(13,17,23,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  [data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
  }

  /* ── Progress bar ── */
  .stProgress > div > div { border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SESSION STATE & MODEL LOADER
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Cache the model so it loads only once."""
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def _resolve_model_path() -> str | None:
    for p in [MODEL_SAVE_PATH, FINAL_MODEL_PATH]:
        if os.path.exists(p):
            return p
    return None


# ──────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────

def run_inference(pil_image: Image.Image,
                  model,
                  threshold: float = THRESHOLD) -> dict:
    """Preprocess PIL image and return prediction dict."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    batch = np.expand_dims(img, axis=0)

    score = float(model.predict(batch, verbose=0)[0][0])
    label = CLASSES[int(score >= threshold)]
    conf  = score if label == "PNEUMONIA" else 1.0 - score
    return {"label": label, "confidence": conf, "score": score}


def compute_gradcam_pil(pil_image: Image.Image, model) -> np.ndarray | None:
    """Compute Grad-CAM overlay and return as RGB uint8 numpy array."""
    import tensorflow as tf

    img_rgb = np.array(pil_image.convert("RGB"))
    img_resized = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    # Find last conv layer (handles sub-models for transfer learning)
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    last_conv = sub.name
                    break
            if last_conv:
                break

    if last_conv is None:
        return None

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            loss = preds[:, 0]
        grads       = tape.gradient(loss, conv_out)
        pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap     = (conv_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
        heatmap     = np.maximum(heatmap, 0)
        heatmap    /= (heatmap.max() + 1e-8)

        h_resized   = cv2.resize(heatmap, IMG_SIZE)
        h_uint8     = np.uint8(255 * h_resized)
        h_colored   = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
        h_colored   = cv2.cvtColor(h_colored, cv2.COLOR_BGR2RGB)

        orig        = cv2.resize(img_rgb, IMG_SIZE)
        overlay     = (orig * 0.55 + h_colored * 0.45).astype(np.uint8)
        return overlay
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.1, max_value=0.9, value=THRESHOLD, step=0.01,
        help="Scores ≥ threshold → PNEUMONIA",
    )

    show_gradcam = st.toggle("Show Grad-CAM Heatmap", value=True)

    st.divider()
    st.markdown("### 📊 Model Performance")

    # Load metrics if available
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        cols = st.columns(2)
        for col, (key, label) in zip(
            [cols[0], cols[1], cols[0], cols[1]],
            [("accuracy","Accuracy"), ("auc","AUC"),
             ("f1_score","F1-Score"), ("recall","Recall")],
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{m[key]:.1%}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("")
    else:
        st.info("Run `python src/evaluate.py` to populate metrics here.")

    st.divider()
    st.markdown("### 📁 Results")
    if os.path.exists(os.path.join(RESULTS_DIR, "confusion_matrix.png")):
        with st.expander("Confusion Matrix"):
            st.image(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    if os.path.exists(os.path.join(RESULTS_DIR, "roc_curve.png")):
        with st.expander("ROC Curve"):
            st.image(os.path.join(RESULTS_DIR, "roc_curve.png"))

    st.divider()
    st.caption("🫁 Pneumonia Detection AI  •  Deep Learning Project")


# ──────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────

st.markdown('<h1 class="hero-title">🫁 Pneumonia Detection AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Upload a chest X-ray — our AI will classify it as '
    '<strong>Normal</strong> or <strong>Pneumonia</strong> in seconds.</p>',
    unsafe_allow_html=True,
)

# ── Check for model ───────────────────────────────────────────
model_path = _resolve_model_path()

if model_path is None:
    st.error(
        "⚠️ **No trained model found.**\n\n"
        "Please train the model first:\n```\npython src/train.py\n```"
    )
    st.stop()

with st.spinner("Loading model..."):
    model = load_model(model_path)

st.success(f"✅ Model loaded: `{os.path.basename(model_path)}`", icon="🧠")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────
tab_single, tab_batch, tab_about = st.tabs(
    ["🔬 Single Prediction", "📂 Batch Prediction", "ℹ️ About"]
)


# ════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ════════════════════════════════════════════════════════

with tab_single:
    uploaded = st.file_uploader(
        "Upload a chest X-ray (JPG / JPEG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    if uploaded is not None:
        pil_image = Image.open(uploaded)

        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("#### 📷 Uploaded X-Ray")
            st.image(pil_image, use_container_width=True, caption=uploaded.name)

        with col_result:
            st.markdown("#### 🧠 Analysis")
            with st.spinner("Running inference..."):
                result = run_inference(pil_image, model, threshold=threshold)

            label = result["label"]
            conf  = result["confidence"]
            score = result["score"]

            # Result badge
            badge_class = "result-pneumonia" if label == "PNEUMONIA" else "result-normal"
            icon        = "🔴" if label == "PNEUMONIA" else "🟢"
            st.markdown(f"""
            <div class="{badge_class}">
              <div class="result-label">{icon} {label}</div>
              <div class="result-conf">Confidence: {conf:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # Confidence bars
            st.markdown("**Probability Distribution**")
            normal_prob = 1.0 - score
            st.markdown(f"🟢 Normal: `{normal_prob:.1%}`")
            st.progress(normal_prob)
            st.markdown(f"🔴 Pneumonia: `{score:.1%}`")
            st.progress(score)

            st.markdown("")
            st.markdown(
                f"**Raw sigmoid score:** `{score:.6f}`  "
                f"*(threshold = {threshold})*"
            )

        # Grad-CAM
        if show_gradcam:
            st.divider()
            st.markdown("#### 🔥 Grad-CAM Attention Heatmap")
            st.caption(
                "Highlights the regions the model focused on when making its decision."
            )
            with st.spinner("Computing Grad-CAM..."):
                overlay = compute_gradcam_pil(pil_image, model)

            if overlay is not None:
                col_orig, col_cam = st.columns(2)
                with col_orig:
                    st.image(pil_image, caption="Original X-Ray", use_container_width=True)
                with col_cam:
                    st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            else:
                st.warning("Grad-CAM is not available for this model architecture.")


# ════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════

with tab_batch:
    st.markdown("Upload multiple X-ray images for bulk classification.")
    batch_files = st.file_uploader(
        "Upload chest X-rays",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.divider()
        results_list = []

        progress_bar = st.progress(0, text="Processing images...")
        for i, f in enumerate(batch_files):
            img = Image.open(f)
            res = run_inference(img, model, threshold=threshold)
            results_list.append({
                "File":       f.name,
                "Prediction": res["label"],
                "Confidence": f"{res['confidence']:.1%}",
                "Score":      round(res["score"], 4),
            })
            progress_bar.progress((i + 1) / len(batch_files),
                                  text=f"Processing {i+1}/{len(batch_files)}...")

        progress_bar.empty()

        # Summary stats
        n_pneu  = sum(1 for r in results_list if r["Prediction"] == "PNEUMONIA")
        n_norm  = len(results_list) - n_pneu

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Images", len(results_list))
        col2.metric("🔴 Pneumonia",  n_pneu)
        col3.metric("🟢 Normal",     n_norm)

        st.divider()
        st.dataframe(results_list, use_container_width=True)

        # Download CSV
        import csv, io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["File", "Prediction", "Confidence", "Score"])
        writer.writeheader()
        writer.writerows(results_list)
        st.download_button(
            "⬇️ Download Results CSV",
            data=buf.getvalue(),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════

with tab_about:
    st.markdown("""
    ## About This Project

    This application uses **deep learning** to classify chest X-ray images as either
    **Normal** or **Pneumonia** — a condition that kills over 2.5 million people annually.

    ### 🏗️ Architecture
    | Component | Details |
    |-----------|---------|
    | Backbone | MobileNetV2 / ResNet50 / EfficientNetB0 |
    | Training Strategy | Two-phase (frozen → fine-tuned) |
    | Input Size | 224 × 224 × 3 (RGB) |
    | Output | Sigmoid probability (0 = Normal, 1 = Pneumonia) |
    | Augmentations | Rotation, shift, shear, zoom, flip, brightness |

    ### 📂 Dataset
    **Chest X-Ray Images (Pneumonia)** — Paul Mooney, Kaggle  
    ~5,800 images • `NORMAL/` and `PNEUMONIA/` classes • Train / Val / Test split

    ### 🔬 Explainability
    **Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights the lung
    regions that most influenced the model's prediction, making the AI interpretable
    to medical professionals.

    ### ⚠️ Disclaimer
    > This tool is for **educational and research purposes only**.
    > It is **not a medical device** and should **not** be used for clinical diagnosis.
    > Always consult a qualified healthcare professional.

    ### 🚀 Quick Start
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Download dataset
    python download_data.py

    # 3. Train the model
    python src/train.py

    # 4. Evaluate
    python src/evaluate.py

    # 5. Single prediction
    python src/predict.py --image data/test/PNEUMONIA/some_image.jpeg

    # 6. Launch this UI
    streamlit run app.py
    ```
    """)
