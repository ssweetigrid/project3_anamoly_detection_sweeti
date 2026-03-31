# 🚦 Overhead Traffic Anomaly Detection 

This repository implements a **Transformer-based anomaly detection system** for surveillance traffic videos.
The project covers **training, concept steering, and robustness evaluation** across multiple experimental setups.

---

# 📌 Project Overview

The goal is to detect unusual traffic events (collisions, congestion, stopped vehicles) using reconstruction-based anomaly detection.

### 🔁 Pipeline

Video Frames
→ ResNet50 Feature Extraction (2048-D)
→ Transformer AE 
→ Reconstruction Error
→ Anomaly Score

---

# ⚙️ Execution Environment

## 🔹 Task 1 (Kaggle GPU)

* Executed on Kaggle (Tesla T4 GPU)
* Includes:

  * Feature extraction
  * Model training (Transformer AE / VAE)
  * ONNX export & optimization
  * Async inference pipeline (latency testing)

---

## 🔹 Tasks 3 & 5 (Local Mac - MPS)

* Executed locally on macOS (Apple Silicon)

* Used:

  * Python virtual environment (`.venv`)
  * MPS acceleration

* Focus:

  * Concept vector steering (Task 3)
  * Robustness analysis (Task 5)

---

## Task 4
* Ablation study is done on diffrent configurations

# 📂 Project Structure

```id="projstruct"
# 📂 Project Structure

```text
.
├── task1/
│   └── anomaly-detection.ipynb        # Task 1 (Kaggle training pipeline)
│
├── task3/
│   └── task3.ipynb                   # Concept steering (no retraining)
│
├── task4/
│   └── task4.ipynb                   # Ablation study (FPS, window, threshold)
│
├── task5/
│   └── task5.ipynb                   # Robustness evaluation
│
├── transformer_model/                # Model architecture code
│
├── app3.py                           # Streamlit app (Task 3 demo)
├── main.ipynb                        # Feature extraction pipeline
├── surveillance_anomaly_detection.ipynb   # Experiments & analysis
│
├── requirements.txt
├── event_labels.txt
└── .gitignore
```

---

# 🧠 Task Descriptions

## ✅ Task 1 — Anomaly Detection (Kaggle)

* Transformer AE  model
* Reconstruction-based anomaly scoring
* Includes:

  * Feature extraction (ResNet50)
  * ONNX export
  * Quantization experiments
  * Async inference pipeline

---

## ✅ Task 3 — Concept Vector Steering (No Retraining)

* Extracts anomaly embeddings
* Uses PCA to compute concept directions
* Applies post-hoc steering using alpha values

### Evaluations:

* AUC-ROC
* Selective anomaly sensitivity
* Calibration metrics

---

## ✅ Task 5 — Robustness Under Corruption

Simulated corruptions:

* Blur
* Noise (rain/snow proxy)
* JPEG compression
* Brightness shift

### Includes:

* AUC degradation analysis
* False positive behavior
* Denoiser recovery experiment
* Ensemble robustness evaluation

---

# 📊 Experiments

Multiple configurations tested:

* Transformer AE (window=16, stride=5)
* Transformer AE (window=8, stride=5)
* Transformer VAE variants

🎯 Target: **80%+ AUC-ROC**

---

# 📦 External Files (Google Drive)

Large files are not included due to GitHub limits.

### 🔗 Add your links here:

* 📁 Dataset: [https://drive.google.com/drive/folders/1_8ZeuLj6iPPfHxfg2lE394H7Eot-AaT1?usp=drive_link]
* 📁 Saved Features: [https://drive.google.com/file/d/1ZkT5x_1DJytPBBA2bu3WNhy--TvnT8as/view?usp=drive_link]
* 📁 Experiments & Models: [https://drive.google.com/drive/folders/1ltgs4neTYXRcMkgTAC4EzJhVtA7eXL0-?usp=drive_link]

### Contains:

* `.npy` feature files
* trained `.pth` models
* experiment outputs

---

# 🛠️ Setup Instructions

```bash id="setupenv"
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

---

# 🚀 Run Project

### Run notebooks

```bash id="runnotebook"
jupyter notebook
```

### Run Streamlit app

```bash id="runapp"
streamlit run app3.py
```

---

# 📈 Reproducibility

All experiments ensure:

* Consistent preprocessing
* Fixed pipeline
* Standard evaluation metrics (AUC-ROC)

---

# ⚠️ Notes

* Large files (`.pth`, `.npy`, experiments) are excluded
* Download external files before running notebooks
* Task 1 outputs originate from Kaggle GPU execution

---

# 👩‍💻 Author

**Sweeti Swami**

---

# ⭐ Highlights

* Transformer-based anomaly detection
* Concept steering without retraining
* Robustness evaluation under corruptions
* Clean modular pipeline


---
