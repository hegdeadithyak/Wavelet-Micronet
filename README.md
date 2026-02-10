# WaveletMicroNet: Ultra-Lightweight Wafer Defect Detection

![Status](https://img.shields.io/badge/Status-Inprogress-orange)
![Platform](https://img.shields.io/badge/Platform-NXP%20i.MX-blue)
![Model Size](https://img.shields.io/badge/Size-1.6MB-green)
![Speed](https://img.shields.io/badge/Speed-213_FPS-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-98.5%25-brightgreen)

> **Competition Track:** AI-enabled Chip Design / Embedded Development  
> **Target:** IESA Vision Summit 2026 - National Hackathon

---

## Summary
Semiconductor fabrication requires zero-error inspection, but traditional cloud-based analysis suffers from high latency and bandwidth bottlenecks. 

**WaveletMicroNet** is a purpose-built Edge AI solution that moves defect detection directly to the manufacturing line. By combining **Discrete Wavelet Transforms (DWT)** with **Quantization Aware Training (QAT)**, we achieve server-grade accuracy on embedded-grade hardware—running at **213 FPS** with a tiny **1.6MB** footprint.

---

## 🏆 Key Performance Indicators (KPIs)

| Metric | WaveletMicroNet (Ours) | Standard ResNet-18 | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | **1.6 MB** | ~45.0 MB | **75x Smaller** |
| **Inference Speed (CPU)** | **213 FPS** | ~15 FPS | **15x Faster** |
| **Accuracy (Test)** | **98.5%** | 98.9% | Comparable |
| **Compute Cost** | **INT8 (Quantized)** | FP32 (Float) | Low Power |

---

## 🧠 The "Micro-Architecture" Innovation

Standard CNNs are often too heavy for edge cameras. We replaced standard heavy layers with signal-processing techniques:

1.  **Wavelet Downsampling (DWT):** Instead of `MaxPooling` (which throws away data), we split images into frequency bands (Low/High). This preserves microscopic defect edges (scratches/particles) that standard pooling blurs out.
2.  **Depthwise Separable Convolutions:** Reduces parameter count by ~9x compared to standard convolutions.
3.  **Quantization Aware Training (QAT):** The model is trained to simulate 8-bit integers, allowing for a **4x size reduction** without the accuracy loss associated with standard post-training quantization.

---

## 📂 Repository Structure

```
.
├── dataset/                       # Wafer defect dataset (train/valid/test)
│   ├── train/                     # Training set (9 defect classes)
│   ├── valid/                     # Validation set
│   └── test/                      # Test set
├── wavelet_qat_out/               # Training outputs
│   ├── best_qat_state.pth         # Best model checkpoint
│   ├── history.csv                # Training metrics log
│   └── model_qat_quantized.pth    # Quantized model
├── train.py                       # Main training script (QAT + Wavelets)
├── test.py                        # Evaluation script with metrics
├── test_onnx.py                   # ONNX model inference test
├── convert_onnx.py                # Convert PyTorch to ONNX
├── plot.py                        # Visualize training history
├── wavelet_model.onnx             # 🚀 FINAL DEPLOYMENT MODEL (0.6MB)
├── requirements.txt               # Python dependencies
└── README.md
```

**Defect Classes (9):** Center, Donut, Clean,Edge-Loc, Edge-Ring, Local, Near-Full, Normal, Random, Scratch

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Train Model (QAT)

Trains the WaveletMicroNet from scratch with Quantization Aware Training:

```bash
python train.py
```

This will:
- Load dataset from `./dataset/`
- Train for 30 epochs (configurable)
- Save best checkpoint to `./wavelet_qat_out/best_qat_state.pth`
- Export training history to `./wavelet_qat_out/history.csv`

### 3. Plot Training Results

Visualize accuracy and loss curves:

```bash
python plot.py
```

Output saved to `plots.png`

### 4. Convert to ONNX

Export the trained PyTorch model for edge deployment:

```bash
python convert_onnx.py
```

Output: `wavelet_model.onnx` (0.6 MB, deployable on NXP i.MX)

### 5. Test ONNX Model

Run inference on test set and benchmark performance:

```bash
python test_onnx.py
```

Expected Output:
```
TEST ACCURACY: 0.985897435897436
Avg inference time per image: 0.0047 sec
FPS: 213.49
```

### 6. Evaluate with Metrics

Generate confusion matrix and classification report:

```bash
python test.py
```

---

## 📋 Dataset Structure

The dataset is organized as ImageFolders with 9 defect classes:

```
dataset/
├── train/
│   ├── Center/
│   ├── Donut/
│   ├── Edge-Loc/
│   ├── Edge-Ring/
│   ├── Local/
│   ├── Near-Full/
│   ├── Clean/
│   ├── Other/
│   └── Scratch/
├── valid/ (same structure)
└── test/ (same structure)
```

---

## 🔧 Configuration

Edit the Python scripts to modify:
- **Image size:** 128×128 (grayscale)
- **Batch size:** 32 (training)
- **Epochs:** 30
- **Learning rate:** Starting at 0.001 with cosine decay
- **Quantization:** INT8 (8-bit) via PyTorch QAT

---

## 📊 Results

- **Test Accuracy:** 98.5% (9-class classification)
- **Model Size:** 1.6 MB (ONNX)
- **Inference Speed:** 213 FPS (CPU, batch=1)
- **ONNX Opset:** 11 (NXP eIQ compatible)

---

## 🏆 Competition Compliance

- **Target Platform:** NXP i.MX (ARM Cortex-A/M)
- **Model Format:** ONNX (Opset 11)
- **Deployment:** Edge inference (no cloud dependency)
- **Benchmark:** Validated on standardized wafer defect dataset

---

## 👥 Team

**Team Name:** HuTLabs
**Contact:** adithyahegdek@gmail.com

