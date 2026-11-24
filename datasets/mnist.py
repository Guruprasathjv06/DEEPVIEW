"""MNIST dataset loader with Stratified 5-Fold Cross-Validation for DEEPVISION."""

import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import torchvision.datasets as datasets
from sklearn.model_selection import StratifiedKFold
from datasets.transforms import get_mnist_transforms

MNIST_CLASSES = [str(i) for i in range(10)]


class TransformedSubset(Dataset):
    """A Dataset wrapper around a base dataset that applies a specific transform."""

    def __init__(self, base_dataset: datasets.MNIST, indices: List[int], transform=None) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        actual_idx = self.indices[idx]
        image, target = self.base_dataset.data[actual_idx], int(self.base_dataset.targets[actual_idx])
        # base_dataset.data is a torch uint8 tensor (28, 28)
        # Convert to PIL Image or apply transform
        image_pil = Image.fromarray(image.numpy(), mode="L")

        if self.transform is not None:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        return image_tensor, target


class MNISTManager:
    """Manages downloading, stratified 5-fold cross-validation splitting,
    and DataLoader creation for MNIST.
    """

    def __init__(
        self,
        data_dir: str = "data",
        n_splits: int = 5,
        seed: int = 42,
        augment_train: bool = True,
    ) -> None:
        """Initialize the MNIST Manager.

        Args:
            data_dir: Directory where raw dataset files are stored.
            n_splits: Number of cross-validation folds (default 5).
            seed: Random seed for reproducible stratified splitting.
            augment_train: Whether to apply data augmentation to training folds.
        """
        self.data_dir = data_dir
        self.n_splits = n_splits
        self.seed = seed
        self.augment_train = augment_train

        os.makedirs(data_dir, exist_ok=True)

        # Download raw training dataset (60,000 samples)
        self.raw_train_set = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=None,
        )

        # Download official test dataset (10,000 samples) - kept strictly separate from CV
        self.raw_test_set = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=get_mnist_transforms(augment=False),
        )

        # Precompute stratified folds on the 60,000 training samples
        self.folds = self._generate_stratified_folds()

    def _generate_stratified_folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified K-fold train/validation indices."""
        targets = self.raw_train_set.targets.numpy()
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        folds = []
        dummy_x = np.zeros(len(targets))
        for train_idx, val_idx in skf.split(dummy_x, targets):
            folds.append((train_idx, val_idx))
        return folds

    def get_fold_loaders(
        self,
        fold_idx: int,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation DataLoaders for a specific cross-validation fold.

        Args:
            fold_idx: Index of the fold (0 to n_splits - 1).
            batch_size: Mini-batch size.
            num_workers: Number of DataLoader subprocess workers.

        Returns:
            Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
        """
        if fold_idx < 0 or fold_idx >= self.n_splits:
            raise ValueError(f"Fold index {fold_idx} out of range [0, {self.n_splits - 1}].")

        train_indices, val_indices = self.folds[fold_idx]

        train_subset = TransformedSubset(
            self.raw_train_set,
            train_indices,
            transform=get_mnist_transforms(augment=self.augment_train),
        )
        val_subset = TransformedSubset(
            self.raw_train_set,
            val_indices,
            transform=get_mnist_transforms(augment=False),
        )

        # Generator for reproducible DataLoader shuffling
        g = torch.Generator()
        g.manual_seed(self.seed + fold_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=g,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        return train_loader, val_loader

    def get_test_loader(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get the DataLoader for the untouched official test set (10,000 samples).

        Args:
            batch_size: Mini-batch size.
            num_workers: Number of DataLoader subprocess workers.

        Returns:
            DataLoader: Official test set loader.
        """
        return DataLoader(
            self.raw_test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    def get_fold_class_distribution(self, fold_idx: int) -> Dict[str, Dict[int, int]]:
        """Verify stratification by returning class counts for a given fold."""
        train_idx, val_idx = self.folds[fold_idx]
        targets = self.raw_train_set.targets.numpy()
        train_targets = targets[train_idx]
        val_targets = targets[val_idx]

        train_counts = {int(k): int(v) for k, v in zip(*np.unique(train_targets, return_counts=True))}
        val_counts = {int(k): int(v) for k, v in zip(*np.unique(val_targets, return_counts=True))}

        return {"train": train_counts, "val": val_counts}
