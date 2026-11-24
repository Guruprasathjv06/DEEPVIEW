"""Performance evaluation metrics for DEEPVISION."""

from typing import Dict, Any, List, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    class_names: List[str] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth target labels (N,).
        y_pred: Predicted class labels (N,) or logit probabilities (N, C).
        class_names: Optional list of class names.

    Returns:
        Dict[str, Any]: Calculated metrics dictionary including accuracy,
                        precision, recall, f1, and confusion matrix.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # If y_pred has logits or probabilities shape (N, C), take argmax
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_dict = {}
    for idx, score in enumerate(per_class_f1):
        name = class_names[idx] if class_names and idx < len(class_names) else str(idx)
        per_class_dict[name] = float(score)

    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "confusion_matrix": cm.tolist(),
        "per_class_f1": per_class_dict,
        "classification_report": report_str,
    }
