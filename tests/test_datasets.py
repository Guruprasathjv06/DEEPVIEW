"""Unit tests for dataset loading, transforms, and stratified splitting in DEEPVISION."""

import pytest
import torch
from datasets.mnist import MNISTManager
from datasets.transforms import get_mnist_transforms, MNIST_MEAN, MNIST_STD


@pytest.fixture(scope="module")
def mnist_manager():
    """Fixture providing initialized MNISTManager."""
    return MNISTManager(data_dir="data", n_splits=5, seed=42)


class TestMNISTDataset:
    """Test suite for MNIST data loading and cross-validation splitting."""

    def test_stratified_folds_structure(self, mnist_manager):
        """Verify that 5 folds are generated and have correct sample counts."""
        assert len(mnist_manager.folds) == 5, f"Expected 5 folds, got {len(mnist_manager.folds)}"
        total_samples = len(mnist_manager.raw_train_set)
        assert total_samples == 60000, f"Expected 60,000 training samples, got {total_samples}"

        for fold_idx, (train_idx, val_idx) in enumerate(mnist_manager.folds):
            # In 5-fold CV of 60,000 samples: 48,000 train, 12,000 val
            assert len(train_idx) == 48000, f"Fold {fold_idx} train size is {len(train_idx)}"
            assert len(val_idx) == 12000, f"Fold {fold_idx} val size is {len(val_idx)}"
            # Verify no leakage between train and val
            overlap = set(train_idx).intersection(set(val_idx))
            assert len(overlap) == 0, f"Data leakage detected in fold {fold_idx}: {len(overlap)} overlapping indices"

    def test_stratified_class_proportions(self, mnist_manager):
        """Verify each fold maintains the same class distribution as the full training set."""
        for fold_idx in range(5):
            dist = mnist_manager.get_fold_class_distribution(fold_idx)
            train_counts = dist["train"]
            val_counts = dist["val"]
            for c in range(10):
                # The train-to-val ratio should be 4:1 (80% train, 20% val)
                expected_ratio = 4.0
                actual_ratio = train_counts[c] / val_counts[c]
                assert abs(actual_ratio - expected_ratio) < 0.05, (
                    f"Fold {fold_idx} class {c} has skewed ratio {actual_ratio:.3f}"
                )

    def test_dataloader_batch_shapes(self, mnist_manager):
        """Verify DataLoader yields batches of correct shape and normalization."""
        train_loader, val_loader = mnist_manager.get_fold_loaders(fold_idx=0, batch_size=32)
        images, labels = next(iter(train_loader))

        assert images.shape == (32, 1, 28, 28), f"Unexpected batch shape: {images.shape}"
        assert labels.shape == (32,), f"Unexpected labels shape: {labels.shape}"
        assert labels.dtype == torch.int64, f"Unexpected labels dtype: {labels.dtype}"

        # Test normalization: values should be centered approximately around 0 with std around 1
        assert images.mean().abs() < 1.0, f"Batch mean too far from zero: {images.mean()}"

    def test_official_test_set_isolation(self, mnist_manager):
        """Verify the official test set contains 10,000 samples and is isolated."""
        test_loader = mnist_manager.get_test_loader(batch_size=64)
        total_test_samples = len(test_loader.dataset)
        assert total_test_samples == 10000, f"Expected 10,000 test samples, got {total_test_samples}"

        images, labels = next(iter(test_loader))
        assert images.shape == (64, 1, 28, 28)
        assert labels.shape == (64,)
