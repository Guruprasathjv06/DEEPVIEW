"""Evaluation pipeline for DEEPVISION."""

from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from evaluation.metrics import compute_metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    class_names: Optional[List[str]] = None,
    collect_samples: bool = True,
    max_samples: int = 32,
) -> Dict[str, Any]:
    """Perform comprehensive model evaluation on a DataLoader.

    Args:
        model: PyTorch neural network.
        dataloader: DataLoader for validation or testing.
        device: Device to run evaluation on.
        criterion: Optional loss function (defaults to CrossEntropyLoss).
        class_names: Optional list of class names.
        collect_samples: Whether to retain sample predictions for visualization.
        max_samples: Maximum number of sample predictions to collect.

    Returns:
        Dict[str, Any]: Evaluation metrics, loss, and sample prediction records.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    model.to(device)

    total_loss = 0.0
    total_samples = 0

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_confs: List[float] = []

    sample_images: List[np.ndarray] = []
    sample_trues: List[int] = []
    sample_preds: List[int] = []
    sample_confs: List[float] = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            confs_np = confs.cpu().numpy()

            all_preds.extend(preds_np)
            all_targets.extend(targets_np)
            all_confs.extend(confs_np)

            # Store sample predictions for visualization
            if collect_samples and len(sample_images) < max_samples:
                needed = max_samples - len(sample_images)
                inputs_np = inputs[:needed].cpu().numpy()
                sample_images.extend([inputs_np[i] for i in range(len(inputs_np))])
                sample_trues.extend(targets_np[:needed])
                sample_preds.extend(preds_np[:needed])
                sample_confs.extend(confs_np[:needed])

    avg_loss = total_loss / max(1, total_samples)

    metrics = compute_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        class_names=class_names,
    )
    metrics["loss"] = float(avg_loss)

    if collect_samples:
        metrics["samples"] = {
            "images": np.array(sample_images),
            "true_labels": np.array(sample_trues),
            "pred_labels": np.array(sample_preds),
            "confidences": np.array(sample_confs),
        }

    return metrics
