# DEEPVISION: CNN-Based Deep Learning Experiments

A complete, modular, and reproducible deep learning experimentation framework built from scratch with Python and PyTorch. DEEPVISION provides rigorous cross-validation, model comparison across architectures, and end-to-end experiment tracking with Weights & Biases (W&B).

---

## Table of Contents

- [Overview](#overview)
- [System Architecture & CUDA Detection](#system-architecture--cuda-detection)
- [Project Structure](#project-structure)
- [CNN Model Architectures](#cnn-model-architectures)
  - [Basic CNN](#basic-cnn)
  - [Improved CNN](#improved-cnn)
- [Datasets & Stratified 5-Fold Cross-Validation](#datasets--stratified-5-fold-cross-validation)
- [Weights & Biases (W&B) Tracking](#weights--biases-wb-tracking)
- [Installation & Setup](#installation--setup)
- [Interactive Web Application (Streamlit)](#interactive-web-application-streamlit)
- [Running Experiments](#running-experiments)
  - [1. Running Automated Unit Tests](#1-running-automated-unit-tests)
  - [2. Quick Pipeline Verification](#2-quick-pipeline-verification)
  - [3. Full Stratified 5-Fold Cross-Validation](#3-full-stratified-5-fold-cross-validation)
  - [4. Custom Execution Options](#4-custom-execution-options)
- [Experiment Outputs & Visualizations](#experiment-outputs--visualizations)
- [Troubleshooting](#troubleshooting)

---

## Overview

DEEPVISION implements a production-grade machine learning workflow for computer vision:
1. **Model Implementations From Scratch**: Clean PyTorch implementations of Basic and Improved CNN architectures with explicit tensor dimensional documentation at every step.
2. **Stratified 5-Fold Cross-Validation**: 60,000 training samples are partitioned into 5 stratified folds with preserved class distributions. A fresh model is initialized for each fold.
3. **Strict Official Test Set Isolation**: The official 10,000-sample test set is strictly sequestered from cross-validation and used exclusively for final evaluation of the selected best model.
4. **Hardware-Aware Execution**: Automatic detection of NVIDIA CUDA GPU acceleration with automatic fallback to optimized multi-threaded CPU execution.
5. **W&B Integration**: Automated experiment tracking, hyperparameter recording, loss/accuracy curves, confusion matrices, prediction sample grids, and checkpoint artifacts (with offline mode support).
6. **Reproducibility**: Global seed management guaranteeing deterministic splits and runs.

---

## System Architecture & CUDA Detection

DEEPVISION automatically queries system hardware upon startup via `training/device.py`:
- Inspects `torch.cuda.is_available()`, device names, device counts, and total VRAM.
- Automatically selects `cuda:0` if an NVIDIA GPU is available.
- Selects `cpu` if no CUDA GPU is detected (e.g. Intel Arc GPU or CPU-only systems).
- Supports manual override via `--device cpu`, `--device cuda`, or `--device auto`.

---

## Project Structure

```
deepvision/
├── configs/                  # Experiment configuration dataclasses
│   ├── __init__.py
│   └── mnist_config.py       # Hyperparameters & settings for MNIST
├── datasets/                 # Dataset managers and transformations
│   ├── __init__.py
│   ├── transforms.py         # Standard normalization & data augmentation
│   └── mnist.py              # MNIST download, stratified 5-fold CV splits
├── models/                   # Neural network architectures
│   ├── __init__.py           # Model factory (get_model)
│   ├── base.py               # BaseCNN class with parameter inspection
│   ├── basic_cnn.py          # Basic CNN (Conv, ReLU, MaxPool, Linear)
│   └── improved_cnn.py       # Improved CNN (BatchNorm, Dropout, 5 Conv layers)
├── training/                 # Model training engine
│   ├── __init__.py
│   ├── device.py             # Automatic CUDA/CPU detection
│   ├── early_stopping.py     # Early stopping monitor
│   ├── reproducibility.py    # Global random seed configuration
│   └── trainer.py            # Epoch loops, validation, checkpoint saving
├── evaluation/               # Metrics and visualization
│   ├── __init__.py
│   ├── metrics.py            # Accuracy, Precision, Recall, F1, Confusion Matrix
│   ├── evaluator.py          # Batch evaluation engine
│   └── visualizer.py         # Plots for learning curves, confusion matrix, predictions
├── experiments/              # Orchestration & tracking
│   ├── __init__.py
│   ├── wandb_tracker.py      # Weights & Biases online/offline tracking
│   ├── cross_validation.py   # Stratified 5-fold cross-validation runner
│   └── run_mnist_experiment.py # Main experiment pipeline CLI
├── tests/                    # Automated pytest test suite
│   ├── __init__.py
│   ├── test_models.py        # Model shapes, parameter counts, forward pass
│   ├── test_datasets.py      # Split ratios, leakage check, normalization
│   └── test_pipeline.py      # Training loop, checkpointing, reproducibility
├── checkpoints/              # Saved model weights (.pt files)
├── outputs/                  # Results, metrics, and plots
│   ├── plots/                # Generated visualizations (PNG)
│   ├── comparison_summary.md # Side-by-side comparison table (Markdown)
│   ├── comparison_summary.csv# Comparison metrics (CSV)
│   └── final_experiment_results.json # Full experiment record (JSON)
├── requirements.txt          # Pinned project dependencies
├── .env.example              # Template for environment variables
├── .gitignore                # Protects credentials and binaries
└── README.md                 # Project documentation
```

---

## CNN Model Architectures

### Basic CNN
Consists of 2 Convolutional blocks, Adaptive Average Pooling, and a 2-layer Fully Connected classifier:

| Layer | Type | Specifications | Output Shape (MNIST) |
|---|---|---|---|
| Input | Image Tensor | 1 channel, 28x28 | `(B, 1, 28, 28)` |
| Conv 1 | Conv2d | 32 filters, 3x3, stride 1, padding 1 | `(B, 32, 28, 28)` |
| Act 1 | ReLU | In-place activation | `(B, 32, 28, 28)` |
| Pool 1 | MaxPool2d | 2x2, stride 2 | `(B, 32, 14, 14)` |
| Conv 2 | Conv2d | 64 filters, 3x3, stride 1, padding 1 | `(B, 64, 14, 14)` |
| Act 2 | ReLU | In-place activation | `(B, 64, 14, 14)` |
| Pool 2 | MaxPool2d | 2x2, stride 2 | `(B, 64, 7, 7)` |
| Adapt | AdaptiveAvgPool2d | (4, 4) spatial resolution | `(B, 64, 4, 4)` |
| Flatten | Flatten | Unroll spatial dimensions | `(B, 1024)` |
| FC 1 | Linear | 1024 -> 128 | `(B, 128)` |
| Act 3 | ReLU | In-place activation | `(B, 128)` |
| FC 2 | Linear | 128 -> 10 (num_classes) | `(B, 10)` |

### Improved CNN
A deeper architecture designed for higher capacity, spatial regularization, and stability:
- **Spatial Batch Normalization (`BatchNorm2d`)** after each convolution layer for faster convergence.
- **Hierarchical Convolutions**: 5 convolutional layers grouped into 3 scale blocks.
- **Dropout Regularization**: Spatial `Dropout2d` (p=0.15–0.25) in convolutional blocks and `Dropout` (p=0.40) in the dense classifier to prevent co-adaptation and overfitting.
- **Dense Batch Normalization (`BatchNorm1d`)** before the classification head.

---

## Datasets & Stratified 5-Fold Cross-Validation

1. **Automatic Download**: MNIST is downloaded via `torchvision.datasets.MNIST` to `data/`.
2. **Preprocessing**:
   - Normalized using official MNIST statistics: $\mu = 0.1307$, $\sigma = 0.3081$.
   - Training folds include subtle geometric augmentation (`RandomRotation(10)` and `RandomCrop(28, padding=2)`).
   - Validation and test folds use deterministic normalization only.
3. **Stratified 5-Fold Partitioning**:
   - The 60,000 training samples are partitioned using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
   - Each fold contains exactly 48,000 training samples and 12,000 validation samples.
   - Every digit class (0–9) is strictly proportional across every fold.
4. **Official Test Set**:
   - The 10,000 official test samples are sequestered and never used during cross-validation.
   - Evaluated only once after all cross-validation runs complete using the highest-performing fold checkpoint.

---

## Weights & Biases (W&B) Tracking

DEEPVISION integrates Weights & Biases for live tracking:
- **Project Name**: `DEEPVISION`
- **Logged Metrics**: Per-epoch `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, and `learning_rate`.
- **Logged Visualizations**:
  - Fold Confusion Matrices (`wandb.Image`)
  - Sample Prediction Grids with True vs Predicted labels and confidences (`wandb.Image`)
- **Artifacts**: Best checkpoint saved per fold and logged to W&B.
- **Offline Mode**: Supported automatically if no internet access is present or if `--wandb-mode offline` is supplied.
- **Security**: The API key is loaded from `.env` via `python-dotenv` and is never printed in logs or committed to version control.

---

## Installation & Setup

### 1. Clone or Open Workspace
```bash
cd c:\Users\GURU PRASATH JV\Desktop\wandb
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your W&B API key:
```env
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=DEEPVISION
WANDB_MODE=online
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Interactive Web Application (Streamlit)

DEEPVISION provides a browser-based deep learning studio with real-time training controls, interactive 5-fold cross-validation analysis, W&B experiment tracking links, and real-time image prediction.

### Launching the Web Application
```bash
streamlit run app.py
```
Or via Python module:
```bash
python -m streamlit run app.py
```

The app starts on `http://localhost:8501` and opens automatically in your default web browser.

### Key Capabilities of the Web Application
1. **Home Dashboard:** High-level project KPIs, detected hardware specs, and interactive architecture flowcharts.
2. **Model & Dataset Inspector:** Interactive layer-by-layer tensor dimension breakdown for Basic CNN and Improved CNN, with random dataset sample previews for MNIST and CIFAR-10.
3. **Interactive Training Engine:** Real-time training with dynamic progress bar, live loss/accuracy curves, metrics cards, and direct checkpoint saving.
4. **5-Fold Cross-Validation Analytics:** Detailed fold-by-fold comparison table with Mean ± Std calculations, confusion matrices, and official untouched test set evaluation.
5. **Real-Time Prediction Engine:** Upload your own digit/object images or draw random test samples to run live PyTorch inference with a 10-class probability distribution chart.
6. **W&B Portal:** Live link to the project dashboard on Weights & Biases with local run history.

---

## Running Experiments

### 1. Running Automated Unit Tests
Run the complete test suite (models, datasets, pipeline, reproducibility):
```bash
pytest -v
```

### 2. Quick Pipeline Verification (1 Epoch)
Verify the entire cross-validation and W&B pipeline across both models:
```bash
python experiments/run_mnist_experiment.py --epochs 1 --batch-size 128 --device cpu --wandb-mode online
```

### 3. Full Stratified 5-Fold Cross-Validation
Run the full 5-epoch cross-validation training on both Basic CNN and Improved CNN:
```bash
python experiments/run_mnist_experiment.py --epochs 5 --batch-size 64 --lr 0.001 --device auto --wandb-mode online
```

### 4. Custom Execution Options
```bash
# Run only Basic CNN
python experiments/run_mnist_experiment.py --models basic --epochs 5

# Run in offline mode (saves local runs without uploading)
python experiments/run_mnist_experiment.py --wandb-mode offline

# Run with SGD optimizer
python experiments/run_mnist_experiment.py --optimizer sgd --lr 0.01
```

---

## Experiment Outputs & Visualizations

Upon experiment completion, all artifacts are saved to `outputs/`:
- `outputs/comparison_summary.md`: Detailed markdown summary and side-by-side metric tables.
- `outputs/comparison_summary.csv`: Machine-readable CSV of cross-validation metrics.
- `outputs/final_experiment_results.json`: Full JSON record of hardware setup, fold metrics, and test evaluation.
- `outputs/plots/`:
  - `learning_curves_foldX.png`: Train vs Validation loss and accuracy curves per fold.
  - `confusion_matrix_foldX.png`: Heatmaps of class prediction distributions.
  - `sample_preds_foldX.png`: Grids of sample digits with predicted vs true labels.
  - `test_evaluation/`: Final confusion matrix and sample prediction grid on official test set.

---

## Troubleshooting

- **Windows Application Control (WinError 4551)**:
  - If Windows Smart App Control blocks unsigned DLLs in Desktop virtualenvs, install dependencies in standard Python user site-packages (`AppData\Local\Programs\Python\...`) using the official PyTorch CPU wheel (`torch==2.5.1+cpu`).
- **Network / Proxy Restrictions**:
  - If internet access is unavailable or proxy blocks W&B servers, set `--wandb-mode offline`. All metrics and plots will be saved locally to disk and can be synced later with `wandb sync wandb/latest-run`.
