"""Evaluation metrics, model evaluation engine, and visualization utilities for DEEPVISION."""

from evaluation.metrics import compute_metrics
from evaluation.evaluator import evaluate_model
from evaluation.visualizer import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_sample_predictions,
)

__all__ = [
    "compute_metrics",
    "evaluate_model",
    "plot_learning_curves",
    "plot_confusion_matrix",
    "plot_sample_predictions",
]
