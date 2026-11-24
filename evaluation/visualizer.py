"""Visualization utilities for training curves, confusion matrices, and predictions."""

import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for headless and CI environments
import matplotlib.pyplot as plt
import seaborn as sns


def plot_learning_curves(
    history: Dict[str, List[float]],
    title_suffix: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation loss and accuracy curves side-by-side.

    Args:
        history: Dict containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        title_suffix: Optional extra string for the figure title.
        save_path: Optional file path to save the generated figure.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Dynamics {title_suffix}".strip(), fontsize=14, fontweight="bold")

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title("Cross-Entropy Loss", fontsize=12)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Accuracy subplot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Acc", color="#d62728", linewidth=2)
    ax2.set_title("Classification Accuracy (%)", fontsize=12)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = False,
) -> plt.Figure:
    """Plot and save a styled confusion matrix heatmap.

    Args:
        cm: Confusion matrix array of shape (C, C).
        class_names: List of class labels.
        title: Title of plot.
        save_path: Optional file path to save the figure.
        normalize: If True, normalize by row (recall per class).

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    if isinstance(cm, list):
        cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(8, 7))

    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_sample_predictions(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    confidences: np.ndarray,
    class_names: List[str],
    n_samples: int = 16,
    title: str = "Sample Model Predictions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a grid of sample test predictions showing true vs predicted labels and probabilities.

    Args:
        images: Array of images with shape (N, C, H, W) or (N, H, W).
        true_labels: Ground truth labels (N,).
        pred_labels: Predicted labels (N,).
        confidences: Prediction confidence probabilities (N,).
        class_names: List of class names.
        n_samples: Number of samples to display (default 16, 4x4 grid).
        title: Title for the grid.
        save_path: Optional path to save the image.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    n_samples = min(n_samples, len(images))
    cols = 4
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.0))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx in range(len(axes)):
        if idx < n_samples:
            img = images[idx]
            # Convert (C, H, W) to (H, W) or (H, W, C)
            if img.ndim == 3:
                if img.shape[0] == 1:
                    img = img.squeeze(0)
                else:
                    img = np.transpose(img, (1, 2, 0))

            true_idx = int(true_labels[idx])
            pred_idx = int(pred_labels[idx])
            true_name = class_names[true_idx] if class_names else str(true_idx)
            pred_name = class_names[pred_idx] if class_names else str(pred_idx)
            conf = confidences[idx] * 100.0

            is_correct = true_idx == pred_idx
            color = "#2ca02c" if is_correct else "#d62728"  # Green if correct, Red if wrong

            if img.ndim == 2:
                axes[idx].imshow(img, cmap="gray")
            else:
                axes[idx].imshow(np.clip(img, 0, 1))

            axes[idx].set_title(
                f"Pred: {pred_name} ({conf:.1f}%)\nTrue: {true_name}",
                color=color,
                fontsize=9,
                fontweight="bold" if not is_correct else "normal",
            )
            # Add colored box border
            for spine in axes[idx].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
