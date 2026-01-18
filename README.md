# HB-BIS - Handwriting-Based Biometric Identification System

**Educational Prototype for Learning Biometric System Design**

> ⚠️ **EDUCATIONAL USE ONLY** - This system is designed for learning and demonstration purposes. It is NOT suitable for real security applications.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Educational Concepts](#educational-concepts)
- [Limitations](#limitations)
- [FAQ](#faq)

---

## 🎯 Overview

HB-BIS is a complete biometric identification system that demonstrates how handwriting analysis can be used for person identification. It implements the full biometric pipeline from image acquisition to decision-making, using both classical machine learning (SVM) and deep learning (SqueezeNet) approaches.

**Key Learning Objectives:**
- Understand the complete biometric system pipeline
- Compare classical vs. deep learning feature extraction
- Learn about similarity metrics and decision thresholds
- Explore biometric database design
- Understand the FAR/FRR trade-off

---

## 📸 Screenshot

Here is a screenshot of HB-BIS in Identification Mode:

![HB-BIS GUI Screenshot](assets/screenshot.png)

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

### Educational Encryption
- XOR-based encryption for images and features
- Demonstrates data protection concepts (NOT secure for production!)

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

## 📖 Usage Guide

### Identification Mode

**Purpose:** Identify who wrote a handwriting sample

**Steps:**
1. Select model (SVM or SqueezeNet)
2. Click "Upload Handwriting Image"
3. Choose an image file
4. Click "Identify User"
5. View results:
   - **Match**: High confidence identification
   - **Uncertain**: Manual review recommended
   - **Unknown**: No match found

**Result Display:**
- User name (if match)
- Distance score (lower = more similar)
- Confidence percentage
- Visual confidence bar

---

### Enrollment Mode

**Purpose:** Register a new user in the system

**Steps:**
1. Enter user name (2-50 characters, alphanumeric + spaces)
2. Click "Upload Handwriting Samples"
3. Select 1-5 images (recommended: 3+)
4. Click "Enroll User"

**Similarity Warning:**
If the new user's handwriting is very similar to an existing user, you'll see a warning. This could indicate:
- Duplicate enrollment (same person enrolling twice)
- Very similar writing styles

You can choose to continue or cancel enrollment.

---

### Model Management

**Statistics Display:**
- Total enrolled users
- Total samples in database
- Average samples per user

**Retrain SqueezeNet:**
Fine-tunes the neural network using triplet loss to better discriminate between enrolled users.

**Requirements:**
- Minimum 3 users
- Minimum 2 samples per user

**Duration:** 2-5 minutes (depends on dataset size and CPU)

---

## 🎓 Educational Concepts

### 1. Feature Extraction Philosophies

**SVM (Classical Approach):**
- Features designed by domain experts
- Each dimension has clear meaning (e.g., "average curvature")
- Interpretable and explainable
- Fast extraction (~20-30ms)
- Works well with small datasets

**SqueezeNet (Deep Learning):**
- Features learned automatically from data
- Dimensions are NOT interpretable
- No manual feature engineering needed
- Slower extraction (~50-100ms on CPU)
- Can be fine-tuned for better accuracy

**Lesson:** Neither approach is universally "better" - each has pros/cons depending on the application!

---

### 2. Biometric Performance Metrics

**False Acceptance Rate (FAR):**
- Probability of accepting an imposter as genuine
- Security metric (lower is better)

**False Rejection Rate (FRR):**
- Probability of rejecting a genuine user
- Usability metric (lower is better)

**The Trade-off:**
- Stricter threshold → Lower FAR, Higher FRR (more secure, less convenient)
- Looser threshold → Higher FAR, Lower FRR (less secure, more convenient)

**Configuration:**
See `config.py` to adjust thresholds:
- `SVM_THRESHOLD_ACCEPT = 0.15`
- `SVM_THRESHOLD_REJECT = 0.30`
- `SQUEEZENET_THRESHOLD_ACCEPT = 0.20`
- `SQUEEZENET_THRESHOLD_REJECT = 0.40`

---

### 3. Distance Metrics

**Cosine Distance:**
- Measures angle between vectors
- Range: [0, 2]
- Good for normalized features
- Used in this system

**Euclidean Distance:**
- Measures straight-line distance
- Sensitive to magnitude
- Alternative metric available in `decision.py`

---

### 4. Preprocessing Pipeline

Each image goes through 5 steps:

1. **Denoising** - Remove scanner noise (Gaussian blur)
2. **Lighting Normalization** - Handle varying illumination (CLAHE)
3. **Binarization** - Separate ink from background (Otsu's method)
4. **Size Normalization** - Standardize to 224×224
5. **Stroke Normalization** - Standardize relative thickness (morphological operations)

**Why?** Raw images vary in quality, lighting, and scale. Preprocessing ensures features capture WRITING STYLE only, not environmental factors.

---

### 5. Encryption (Educational)

**What's Implemented:**
- XOR cipher with fixed key
- Base64 encoding for text-safe storage

**Why It's NOT Secure:**
- Fixed key hardcoded in source
- Vulnerable to known-plaintext attacks
- No authentication (can't detect tampering)
- No initialization vector (patterns visible)

**Production Alternatives:**
- AES-256-GCM (symmetric)
- RSA-2048+ (asymmetric)
- Proper key derivation (PBKDF2, Argon2)
- Secure key storage (HSM, TPM)

**Purpose:** Demonstrate the CONCEPT of data protection for learning!

---

## ⚠️ Limitations

This is an **educational prototype** with intentional limitations:

### Security
- ❌ XOR encryption provides NO real security
- ❌ No liveness detection (vulnerable to photo attacks)
- ❌ No secure authentication or access control
- ❌ Unaudited code

### Scalability
- ❌ Not optimized for large databases (100+ users)
- ❌ Linear search (no indexing)
- ❌ File-based storage (not SQL/NoSQL)

### Accuracy
- ❌ Thresholds not professionally calibrated
- ❌ Limited to handwriting (no multi-modal biometrics)
- ❌ No quality score for samples

### Use Cases
✅ **Appropriate for:**
- Educational demonstrations
- Learning biometric concepts
- CV/portfolio projects
- Research prototypes

❌ **NOT appropriate for:**
- Real security systems
- Production authentication
- Financial applications
- Healthcare or government use

---

## ❓ FAQ

**Q: Can I use this for my company's authentication system?**  
A: **NO.** This is educational only. For production systems, use professional biometric solutions with proper security audits.

**Q: Why does enrollment require multiple samples?**  
A: Multiple samples create a more robust template that accounts for natural variation in handwriting. This reduces false rejections.

**Q: What's the difference between identification and verification?**  
A:
- **Identification (1:N)**: "Who is this?" - Compare against all enrolled users
- **Verification (1:1)**: "Is this person X?" - Compare against one specific user

This system implements **identification**.

**Q: Why does SqueezeNet need retraining?**  
A: Pre-trained SqueezeNet has never seen handwriting data. Fine-tuning with triplet loss teaches it to discriminate between YOUR specific users, improving accuracy.

**Q: Can I add my own feature extraction methods?**  
A: Yes! Edit `layers/feature_engineering.py`. Follow the pattern:
1. Create feature extraction function
2. Return numpy array
3. Update `config.py` with feature dimension
4. Modify GUI to include new model option

**Q: What image quality works best?**  
A: For best results:
- High contrast (dark ink, light paper)
- 300+ DPI resolution
- Minimal background noise
- Consistent writing surface

**Q: How do I reset the database?**  
A: Delete the `database/` folder. It will be recreated on next use.

---

## 🔬 For Researchers & Students

### Experiment Ideas

1. **Threshold Tuning**: Adjust thresholds in `config.py` and observe FAR/FRR changes
2. **Feature Engineering**: Design new handcrafted features in `feature_engineering.py`
3. **Data Augmentation**: Add rotation, scaling, or noise to training samples
4. **Multi-Writer Dataset**: Test with IAM Handwriting Database
5. **Comparison Study**: Compare SVM vs SqueezeNet accuracy on same dataset
6. **Preprocessing Impact**: Disable individual preprocessing steps and measure accuracy drop

### Extending the System

**Add New Capabilities:**
- Verification mode (1:1 matching)
- Confidence score calibration
- Rejection threshold optimization
- Feature visualization (t-SNE plots)
- Audit logging
- Multi-language support

**Integrate with Research:**
- Replace SqueezeNet with ResNet/EfficientNet
- Implement Siamese networks
- Try metric learning approaches
- Add signature verification

---

## 📚 References & Learning Resources

**Biometric Systems:**
- Jain, A. K., et al. "Biometrics: A Tool for Information Security" (2006)
- Ross, A., et al. "Handbook of Biometrics" (2007)

**Handwriting Analysis:**
- Plamondon, R., & Srihari, S. N. "Online and off-line handwriting recognition: A comprehensive survey" (2000)

**Deep Learning for Biometrics:**
- Parkhi, O. M., et al. "Deep Face Recognition" (2015)
- Schroff, F., et al. "FaceNet: A Unified Embedding for Face Recognition" (Triplet Loss, 2015)

**SqueezeNet:**
- Iandola, F. N., et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters" (2016)

---

## 📄 License

This is an educational prototype provided "as-is" for learning purposes.

**NOT licensed for production use.**

---

## 🙏 Acknowledgments

Built as an educational demonstration of biometric system design principles.

Special thanks to the open-source community for:
- PyTorch and torchvision
- OpenCV
- scikit-learn
- NumPy/SciPy

---

## 📧 Support

This is an educational project. For questions:
1. Review this README carefully
2. Check the code comments (extensively documented)
3. Run the demonstration modes in each module

**Remember:** This system is designed to teach concepts, not for production use!

---

**Happy Learning! 🚀**
