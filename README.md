# Handwriting-Based Biometric Identification System

---

## 🎯 Overview

HB-BIS is a complete biometric identification system that demonstrates how handwriting analysis can be used for person identification. It implements the full biometric pipeline from image acquisition to decision-making, using both classical machine learning (SVM) and deep learning (SqueezeNet) approaches.

---

## ✨ Features

### Dual-Model Approach

- **SVM Model**: 42-dimensional handcrafted features (ink density, stroke orientation, curvature, geometry, spacing, statistical moments)

- **SqueezeNet Model**: 512-dimensional deep learned embeddings with optional triplet loss fine-tuning

### Complete Biometric Pipeline
1. **Data Acquisition**: Image loading with quality validation
2. **Preprocessing**: Denoising, lighting normalization, binarization, size/stroke normalization
3. **Feature Engineering**: Automated feature extraction (SVM or SqueezeNet)
4. **Database**: Encrypted storage for images and features
5. **Decision Layer**: Similarity-based identification with configurable thresholds
6. **GUI**: User-friendly Tkinter interface

### Encryption
- XOR-based encryption for images and features
- Demonstrates data protection concepts 

### Model Retraining
- Manual retraining of SqueezeNet using triplet loss
- Minimum 3 users with 2+ samples each required

---

## 🏗️ System Architecture

```
HB-BIS/
├── main.py                    # Entry point
├── config.py                  # Configuration and thresholds
├── requirements.txt           # Dependencies
│
├── layers/                    # 6 Modular Layers
│   ├── data_acquisition.py   # Layer 1: Image I/O & encryption
│   ├── preprocessing.py       # Layer 2: Image enhancement & normalization
│   ├── feature_engineering.py # Layer 3: SVM & SqueezeNet features
│   ├── database.py            # Layer 4: Encrypted storage
│   ├── decision.py            # Layer 5: Similarity & decisions
│   └── gui.py                 # Layer 6: User interface
│
├── models/                    # Model wrappers
│   └── squeezenet_model.py    # Retraining with triplet loss
│
├── utils/                     # Utilities
│   ├── encryption.py          # Educational XOR encryption
│   └── validators.py          # Input validation
│
└── database/                  # Created at runtime
    ├── metadata.json
    └── users/
        └── {user_id}/
            ├── raw_images/
            ├── svm_features/
            └── squeezenet_embeddings/
```

---

## 📦 Installation

### Requirements
- Python 3.7+
- 2GB free disk space (for model download)
- Internet connection (first run only)

### Step 1: Clone or Download
```bash
cd HB-BIS
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies:**
- numpy, scipy
- opencv-python, Pillow, scikit-image
- torch, torchvision
- scikit-learn
- matplotlib

**Note:** Tkinter comes built-in with Python (no installation needed)

### Step 3: Verify Installation
```bash
python main.py
```

---

## 🚀 Quick Start

### Running the System
```bash
python main.py
```

This will:
1. Check dependencies
2. Download SqueezeNet model (first run only, ~5MB)
3. Launch the GUI

### First Steps

1. **Enroll a User**
   - Enter user name
   - Upload 1-5 handwriting samples (JPEG/PNG)
   - Click "Enroll User"

2. **Identify a User**
   - Upload a handwriting sample
   - Select model (SVM or SqueezeNet)
   - Click "Identify User"
   - View results: Match/Uncertain/Unknown

3. **Manage System**
   - View database statistics
   - Retrain SqueezeNet (requires ≥3 users with ≥2 samples each)
   
---
