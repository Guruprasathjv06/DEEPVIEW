"""CIFAR-10 dataset manager with Stratified 5-Fold Cross-Validation for DEEPVISION."""

import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.datasets as datasets
from sklearn.model_selection import StratifiedKFold
from datasets.transforms import get_cifar10_transforms

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class TransformedCIFARSubset(Dataset):
    """Dataset wrapper for CIFAR-10 that applies specific transform per split."""

    def __init__(self, base_dataset: datasets.CIFAR10, indices: List[int], transform=None) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        actual_idx = self.indices[idx]
        image, target = self.base_dataset.data[actual_idx], int(self.base_dataset.targets[actual_idx])
        # image is numpy array (H, W, 3) in uint8
        image_pil = Image.fromarray(image)

        if self.transform is not None:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image_tensor, target


class CIFAR10Manager:
    """Manages downloading, stratified 5-fold cross-validation splitting,
    and DataLoader creation for CIFAR-10.
    """

    def __init__(
        self,
        data_dir: str = "data",
        n_splits: int = 5,
        seed: int = 42,
        augment_train: bool = True,
    ) -> None:
        """Initialize the CIFAR-10 Manager.

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

        # Download raw training dataset (50,000 samples)
        self.raw_train_set = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=None,
        )

        # Download official test dataset (10,000 samples) - kept strictly separate from CV
        self.raw_test_set = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=get_cifar10_transforms(augment=False),
        )

        # Precompute stratified folds on the 50,000 training samples
        self.folds = self._generate_stratified_folds()

    def _generate_stratified_folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified K-fold train/validation indices."""
        targets = np.array(self.raw_train_set.targets)
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

        train_subset = TransformedCIFARSubset(
            self.raw_train_set,
            train_indices,
            transform=get_cifar10_transforms(augment=self.augment_train),
        )
        val_subset = TransformedCIFARSubset(
            self.raw_train_set,
            val_indices,
            transform=get_cifar10_transforms(augment=False),
        )

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
