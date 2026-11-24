"""Improved CNN architecture for DEEPVISION.

Architecture Pipeline & Tensor Dimensions:
    Input: (B, in_channels, H, W)   [For MNIST: (B, 1, 28, 28)]

    Block 1 (Low-level features):
        - Conv2d(in_channels, 32, kernel_size=3, padding=1) -> (B, 32, 28, 28)
        - BatchNorm2d(32) -> (B, 32, 28, 28)
        - ReLU() -> (B, 32, 28, 28)
        - Conv2d(32, 32, kernel_size=3, padding=1) -> (B, 32, 28, 28)
        - BatchNorm2d(32) -> (B, 32, 28, 28)
        - ReLU() -> (B, 32, 28, 28)
        - MaxPool2d(2, 2) -> (B, 32, 14, 14)
        - Dropout2d(p=0.15) -> (B, 32, 14, 14)

    Block 2 (Mid-level features):
        - Conv2d(32, 64, kernel_size=3, padding=1) -> (B, 64, 14, 14)
        - BatchNorm2d(64) -> (B, 64, 14, 14)
        - ReLU() -> (B, 64, 14, 14)
        - Conv2d(64, 64, kernel_size=3, padding=1) -> (B, 64, 14, 14)
        - BatchNorm2d(64) -> (B, 64, 14, 14)
        - ReLU() -> (B, 64, 14, 14)
        - MaxPool2d(2, 2) -> (B, 64, 7, 7)
        - Dropout2d(p=0.25) -> (B, 64, 7, 7)

    Block 3 (High-level abstract features):
        - Conv2d(64, 128, kernel_size=3, padding=1) -> (B, 128, 7, 7)
        - BatchNorm2d(128) -> (B, 128, 7, 7)
        - ReLU() -> (B, 128, 7, 7)
        - AdaptiveAvgPool2d((3, 3)) -> (B, 128, 3, 3)
        - Dropout2d(p=0.25) -> (B, 128, 3, 3)

    Classifier:
        - Flatten() -> (B, 128 * 3 * 3) = (B, 1152)
        - Linear(1152, 256) -> (B, 256)
        - BatchNorm1d(256) -> (B, 256)
        - ReLU() -> (B, 256)
        - Dropout(p=0.40) -> (B, 256)
        - Linear(256, num_classes) -> (B, num_classes) [For MNIST: (B, 10)]
"""

import torch
import torch.nn as nn
from models.base import BaseCNN


class ImprovedCNN(BaseCNN):
    """Improved CNN Architecture with:
    - 5 Convolutional layers structured into 3 hierarchical blocks
    - Spatial Batch Normalization after every convolution
    - Dropout regularization (both spatial 2D dropout and fully connected dropout)
    - Fully connected classifier with Batch Normalization and Dropout
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout_conv: float = 0.20,
        dropout_fc: float = 0.40,
    ) -> None:
        """Initialize the Improved CNN.

        Args:
            in_channels: Input channels (1 for MNIST, 3 for CIFAR-10).
            num_classes: Output classes (default 10).
            dropout_conv: Dropout probability for convolutional blocks.
            dropout_fc: Dropout probability for classifier dense layers.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Hierarchical Feature Extractor
        self.features = nn.Sequential(
            # Block 1: 32 channels
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv * 0.75),

            # Block 2: 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv),

            # Block 3: 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Dropout2d(p=dropout_conv),
        )

        # Regularized Dense Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Improved CNN.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits
