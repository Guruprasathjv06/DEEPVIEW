"""Unit tests for the training pipeline, checkpointing, and reproducibility in DEEPVISION."""

import os
import shutil
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import BasicCNN, ImprovedCNN
from training.trainer import ModelTrainer
from training.device import get_device
from training.reproducibility import set_seed
from evaluation.metrics import compute_metrics


class TestPipeline:
    """Test suite for training pipeline and reproducibility."""

    @pytest.fixture
    def synthetic_data(self):
        """Create a tiny synthetic dataset for pipeline validation."""
        x = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        return loader

    def test_single_epoch_training_step(self, synthetic_data, tmp_path):
        """Verify that training and validation run for 1 epoch and save checkpoints."""
        device = get_device("cpu")
        model = BasicCNN(in_channels=1, num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        checkpoint_dir = str(tmp_path / "checkpoints")

        trainer = ModelTrainer(
            model=model,
            device=device,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            model_name="test_cnn",
        )

        results = trainer.fit(
            train_loader=synthetic_data,
            val_loader=synthetic_data,
            epochs=1,
        )

        assert len(results["history"]["train_loss"]) == 1
        assert len(results["history"]["val_loss"]) == 1
        assert results["history"]["train_loss"][0] > 0
        assert 0.0 <= results["history"]["train_acc"][0] <= 1.0

        # Verify best and latest checkpoint exist
        best_ckpt = os.path.join(checkpoint_dir, "test_cnn_best.pt")
        latest_ckpt = os.path.join(checkpoint_dir, "test_cnn_latest.pt")
        assert os.path.exists(best_ckpt), "Best checkpoint was not created"
        assert os.path.exists(latest_ckpt), "Latest checkpoint was not created"

        # Verify loading checkpoint
        loaded = torch.load(best_ckpt, map_location=device, weights_only=False)
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded
        assert loaded["epoch"] == 1

    def test_reproducibility(self, synthetic_data):
        """Verify that fixed seeds produce identical training loss."""
        device = get_device("cpu")

        # Run 1
        set_seed(42)
        model1 = BasicCNN(in_channels=1, num_classes=10)
        opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        trainer1 = ModelTrainer(model=model1, device=device, optimizer=opt1)
        res1 = trainer1.train_one_epoch(synthetic_data)

        # Run 2
        set_seed(42)
        model2 = BasicCNN(in_channels=1, num_classes=10)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        trainer2 = ModelTrainer(model=model2, device=device, optimizer=opt2)
        res2 = trainer2.train_one_epoch(synthetic_data)

        assert abs(res1["loss"] - res2["loss"]) < 1e-6, "Losses differ despite fixed seed!"
        assert res1["accuracy"] == res2["accuracy"], "Accuracies differ despite fixed seed!"

    def test_metric_computation(self):
        """Verify precision, recall, f1, accuracy metrics on deterministic values."""
        y_true = torch.tensor([0, 1, 2, 2, 0, 1])
        y_pred = torch.tensor([0, 1, 2, 1, 0, 1])  # 5 out of 6 correct

        metrics = compute_metrics(y_true, y_pred)
        assert abs(metrics["accuracy"] - (5 / 6)) < 1e-5
        assert metrics["f1_macro"] > 0
        assert metrics["precision_macro"] > 0
        assert metrics["recall_macro"] > 0
        assert "confusion_matrix" in metrics
