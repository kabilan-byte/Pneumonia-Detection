# 🫁 Pneumonia Detection — Deep Learning + OpenCV

> **AI-powered chest X-ray classification: NORMAL vs PNEUMONIA**  
> *மார்பு X-ray images-ஐ வைத்து Pneumonia detect பண்ணும் Deep Learning project*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📖 Project Description (திட்ட விளக்கம்)

This project uses **Convolutional Neural Networks (CNN)** and **Transfer Learning** (MobileNetV2 / ResNet50) to classify chest X-ray images into:

| Class | Label | Meaning |
|-------|-------|---------|
| 🟢 NORMAL | 0 | Healthy lungs |
| 🔴 PNEUMONIA | 1 | Lung infection detected |

**Key Features:**
- ✅ Deep Learning with TensorFlow/Keras
- ✅ OpenCV preprocessing (resize, normalize, CLAHE enhancement)
- ✅ Data Augmentation (rotation, flip, zoom, shift)
- ✅ Two-phase Transfer Learning (frozen → fine-tune)
- ✅ Grad-CAM heatmap visualization
- ✅ Streamlit web app with dark theme
- ✅ Beginner-friendly code with Tamil + English comments

---

## 📂 Project Folder Structure (கோப்பகட்டமைப்பு)

```
Pneumonia Detection/
│
├── 📂 src/                     ← All Python source files (code files)
│   ├── __init__.py             ← Makes src/ a Python package
│   ├── train.py                ← Model training script
│   ├── test.py                 ← Model evaluation on test set
│   ├── main.py                 ← Single image prediction (beginner-friendly)
│   ├── predict.py              ← Advanced prediction + Grad-CAM
│   ├── evaluate.py             ← Detailed evaluation with plots
│   ├── utils.py                ← Shared helper functions
│   ├── preprocess.py           ← OpenCV image preprocessing
│   ├── augmentation.py         ← Data augmentation generators
│   ├── model_cnn.py            ← Custom CNN architecture
│   └── model_transfer.py       ← MobileNetV2 / ResNet50 transfer models
│
├── 📂 data/                    ← Dataset folder (gitignored)
│   ├── train/
│   │   ├── NORMAL/             ← Normal chest X-rays
│   │   └── PNEUMONIA/          ← Pneumonia chest X-rays
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── 📂 models/                  ← Saved trained models
│   ├── best_model.h5           ← Best validation accuracy model
│   └── final_model.h5          ← Final epoch model
│
├── 📂 results/                 ← All output plots and metrics
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics_chart.png
│   ├── training_curves_phase1.png
│   ├── training_curves_phase2.png
│   └── metrics.json
│
├── app.py                      ← Streamlit web application
├── config.py                   ← Central configuration (paths, hyperparameters)
├── download_data.py            ← Kaggle dataset download script
├── requirements.txt            ← Python package dependencies
├── .gitignore                  ← Git ignore rules
└── README.md                   ← This file
```

---

## ⚙️ Setup Instructions (அமைப்பு வழிமுறைகள்)

### Prerequisites (தேவையானவை)
- Python 3.9 or higher
- pip (Python package manager)
- Kaggle account (for dataset download)

---

### Step 1 — Create Virtual Environment (Virtual Environment உருவாக்கு)

> **ஏன் venv தேவை?** — packages globally install ஆகாமல், இந்த project-க்கு மட்டும் isolated environment உருவாக்க.  
> *(Why venv? To keep project packages isolated from your system Python.)*

```bash
# macOS / Linux
python3 -m venv venv

# Windows
python -m venv venv
```

---

### Step 2 — Activate Virtual Environment (Activate பண்ணு)

```bash
# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

> ✅ You'll see `(venv)` prefix in your terminal when activated.  
> *(Terminal-ல் `(venv)` தெரிந்தால் activate ஆகிவிட்டது.)*

---

### Step 3 — Install Dependencies (Packages install பண்ணு)

```bash
pip install -r requirements.txt
```

> இது `requirements.txt`-ல் உள்ள அனைத்து packages-ஐயும் install பண்ணும்.  
> *(This installs all required packages listed in requirements.txt.)*

---

### Step 4 — Download Dataset (Dataset download பண்ணு)

**Option A: Kaggle CLI (Recommended)**

1. Create a Kaggle account at [kaggle.com](https://kaggle.com)
2. Go to Account → API → Create New Token → Download `kaggle.json`
3. Place `kaggle.json` in `~/.kaggle/` (macOS/Linux) or `%USERPROFILE%\.kaggle\` (Windows)
4. Run:

```bash
python download_data.py
```

**Option B: Manual Download**

1. Go to [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Download and extract
3. Organize into `data/train/`, `data/val/`, `data/test/` folders

---

## 🚀 Running the Project (Project run பண்ணு)

### 🔵 Training (Model Train பண்ணு)

```bash
# Default (MobileNetV2, 30 epochs)
python src/train.py

# Custom model type
python src/train.py --model mobilenetv2    # MobileNetV2 (recommended)
python src/train.py --model resnet50       # ResNet50
python src/train.py --model cnn            # Custom CNN

# Full custom options
python src/train.py --model mobilenetv2 --epochs 50 --batch-size 32 --lr 0.0001

# Skip fine-tuning phase
python src/train.py --no-fine-tune
```

**What happens during training:**
1. 🟢 Phase 1: Train classification head (base model frozen)
2. 🟠 Phase 2: Fine-tune top layers with lower learning rate
3. 📈 Saves training curves to `results/`
4. 💾 Saves best model to `models/best_model.h5`

---

### 🔴 Testing / Evaluation (Model Test பண்ணு)

```bash
# Evaluate on test set (auto-detects best model)
python src/test.py

# Specify model explicitly
python src/test.py --model models/best_model.h5

# Custom threshold and batch size
python src/test.py --threshold 0.45 --batch-size 16
```

**Outputs:**
- `results/confusion_matrix.png` — True vs Predicted counts
- `results/roc_curve.png` — ROC curve with AUC score
- `results/metrics_chart.png` — Bar chart of all metrics
- `results/metrics.json` — JSON with all numeric metrics

---

### 🟢 Single Image Prediction (ஒரு image predict பண்ணு)

```bash
# Basic prediction
python src/main.py --image data/test/PNEUMONIA/person1_virus_6.jpeg

# Custom options
python src/main.py --image xray.jpg --threshold 0.4
python src/main.py --image xray.jpg --no-gradcam     # Skip Grad-CAM
python src/main.py --image xray.jpg --no-save        # Don't save output
python src/main.py --image xray.jpg --model models/final_model.h5
```

**Output Example:**
```
╔══════════════════════════════════════════════════════════════╗
║               🫁  Pneumonia Detection — Prediction           ║
╚══════════════════════════════════════════════════════════════╝

🖼️   Image    : data/test/PNEUMONIA/person1_virus_6.jpeg
📐  Input shape: (1, 224, 224, 3)

══════════════════════════════════════════════════════════
  🫁  PREDICTION RESULT
══════════════════════════════════════════════════════════
  Image       : person1_virus_6.jpeg
  Prediction  : 🔴  PNEUMONIA
  Confidence  : 97.43%
  Raw Score   : 0.974300  (threshold = 0.5)
  Meaning     : Pneumonia detected! ⚕️
══════════════════════════════════════════════════════════
```

---

### 🌐 Web Application (Streamlit App)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Features:**
- 🖼️ Drag & drop X-ray upload
- 🧠 Real-time prediction with confidence gauge
- 🔥 Grad-CAM heatmap overlay
- 📂 Batch prediction (multiple images)
- 📊 Model metrics in sidebar

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~95% |
| Precision | ~94% |
| Recall | ~97% |
| F1-Score | ~95% |
| AUC | ~99% |

> *Actual results depend on training duration, model type, and dataset size.*

---

## 🏗️ Architecture Overview

```
Input Image (224×224×3)
        ↓
  OpenCV Preprocessing
  (resize, normalize, BGR→RGB)
        ↓
  Data Augmentation
  (rotation, flip, zoom, shift)
        ↓
  MobileNetV2 (frozen base)
        ↓
  Global Average Pooling
        ↓
  Dense(256, ReLU) + Dropout(0.5)
        ↓
  Dense(1, Sigmoid)
        ↓
  Output: 0.0 (NORMAL) ←→ 1.0 (PNEUMONIA)
```

---

## 🔧 Configuration (config.py)

All hyperparameters are in `config.py`. Edit once, affects all scripts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_TYPE` | `"mobilenetv2"` | Architecture choice |
| `IMG_SIZE` | `(224, 224)` | Input image dimensions |
| `BATCH_SIZE` | `32` | Training batch size |
| `EPOCHS` | `30` | Max training epochs |
| `LEARNING_RATE` | `1e-4` | Phase 1 learning rate |
| `FINE_TUNE_LR` | `1e-5` | Phase 2 fine-tuning LR |
| `THRESHOLD` | `0.5` | Prediction threshold |

---

## 📦 All Terminal Commands Summary

```bash
# ── Setup ────────────────────────────────────────────
python3 -m venv venv              # Create virtual environment
source venv/bin/activate          # Activate (macOS/Linux)
pip install -r requirements.txt   # Install packages

# ── Data ─────────────────────────────────────────────
python download_data.py           # Download dataset from Kaggle

# ── Training ─────────────────────────────────────────
python src/train.py               # Train with defaults
python src/train.py --model cnn   # Train custom CNN

# ── Testing ──────────────────────────────────────────
python src/test.py                # Evaluate on test set

# ── Prediction ───────────────────────────────────────
python src/main.py --image path/to/xray.jpg

# ── Web App ──────────────────────────────────────────
streamlit run app.py

# ── Deactivate venv ───────────────────────────────────
deactivate
```

---

## ⚠️ Medical Disclaimer (முக்கிய குறிப்பு)

> **This tool is for educational and research purposes ONLY.**  
> **இது கல்வி மற்றும் ஆராய்ச்சி நோக்கங்களுக்காக மட்டுமே.**  
>
> This AI model is **NOT a medical device** and should **NOT** be used for clinical diagnosis. Always consult a qualified healthcare professional for medical advice.  
>
> *(இந்த AI model ஒரு medical device அல்ல. எந்த நோயையும் diagnose பண்ண இதை மட்டும் நம்பாதீர்கள். எப்போதும் ஒரு certified doctor-ஐ consult பண்ணுங்கள்.)*

---

## 📚 Dataset

**Chest X-Ray Images (Pneumonia)**  
Source: [Kaggle — Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
~5,800 images | JPEG format | NORMAL & PNEUMONIA classes

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add: my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — Free for educational use.

---

*Made with ❤️ for learning Deep Learning & Medical AI*
