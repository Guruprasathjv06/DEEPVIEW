"""Early stopping callback for PyTorch training."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """Early stops the training if monitored metric doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        """Initialize EarlyStopping.

        Args:
            patience: How many epochs to wait after last improvement before stopping.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            mode: 'min' for loss monitoring, 'max' for accuracy monitoring.
            verbose: If True, prints a message for each improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def step(self, current_value: float) -> bool:
        """Evaluate current metric and determine whether to stop training.

        Args:
            current_value: Latest value of the monitored metric.

        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """
        score = -current_value if self.mode == "min" else current_value

        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False
