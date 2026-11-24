"""Model trainer for DEEPVISION."""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles end-to-end training, validation, checkpointing, and metric logging."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model",
        fold_idx: Optional[int] = None,
        epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> None:
        """Initialize ModelTrainer.

        Args:
            model: PyTorch model to train.
            device: Runtime torch.device (CPU or CUDA).
            optimizer: PyTorch optimizer instance.
            criterion: Loss function (defaults to CrossEntropyLoss).
            scheduler: Optional learning rate scheduler.
            checkpoint_dir: Directory where checkpoints are saved.
            model_name: Base identifier for checkpoint filenames.
            fold_idx: Optional fold number if running cross-validation.
            epoch_callback: Optional callback invoked after each epoch with metrics.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.fold_idx = fold_idx
        self.epoch_callback = epoch_callback

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rate": [],
        }

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Execute one complete training epoch over train_loader.

        Returns:
            Dict[str, float]: Training loss and accuracy for this epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)

        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Execute validation pass over val_loader without gradient computation.

        Returns:
            Dict[str, float]: Validation loss and accuracy for this epoch.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_loss = running_loss / max(1, total)
        val_acc = correct / max(1, total)

        return {"loss": val_loss, "accuracy": val_acc}

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool = False,
    ) -> str:
        """Save model checkpoint to disk.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss.
            val_acc: Validation accuracy.
            is_best: Whether this is the best performing checkpoint so far.

        Returns:
            str: Path to saved checkpoint file.
        """
        tag = f"{self.model_name}_fold{self.fold_idx}" if self.fold_idx is not None else self.model_name
        filename = f"{tag}_best.pt" if is_best else f"{tag}_latest.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint_data = {
            "epoch": epoch,
            "model_name": self.model_name,
            "fold_idx": self.fold_idx,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "history": self.history,
        }

        torch.save(checkpoint_data, filepath)
        return filepath

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> Dict[str, Any]:
        """Full training loop across all epochs.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Total number of training epochs.
            early_stopping: Optional EarlyStopping callback.

        Returns:
            Dict[str, Any]: Complete training results including histories and best metrics.
        """
        logger.info(
            f"Starting training: {self.model_name} "
            f"(Fold: {self.fold_idx if self.fold_idx is not None else 'N/A'}, Epochs: {epochs}, Device: {self.device})"
        )

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["learning_rate"].append(current_lr)

            elapsed = time.time() - t0

            # Check if current epoch is the best (based on validation loss)
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_acc = val_metrics["accuracy"]
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics["loss"], val_metrics["accuracy"], is_best=True)

            # Always save latest checkpoint
            self.save_checkpoint(epoch, val_metrics["loss"], val_metrics["accuracy"], is_best=False)

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']*100:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                f"LR: {current_lr:.6f} | Time: {elapsed:.2f}s {'*' if is_best else ''}"
            )

            # Invoke external callback (e.g. W&B logging)
            if self.epoch_callback:
                step_metrics = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "learning_rate": current_lr,
                }
                self.epoch_callback(epoch, step_metrics)

            # Early stopping check
            if early_stopping and early_stopping.step(val_metrics["loss"]):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        total_elapsed = time.time() - start_time
        logger.info(
            f"Training completed in {total_elapsed:.2f}s. "
            f"Best Val Loss: {self.best_val_loss:.4f}, Best Val Acc: {self.best_val_acc*100:.2f}% (Epoch {self.best_epoch})"
        )

        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time_seconds": total_elapsed,
        }
