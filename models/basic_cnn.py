"""Basic CNN architecture for DEEPVISION.

Architecture Pipeline:
    Input: (B, in_channels, H, W)   [For MNIST: (B, 1, 28, 28)]
    Layer 1:
        - Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1) -> Output: (B, 32, 28, 28)
        - ReLU() -> Output: (B, 32, 28, 28)
        - MaxPool2d(kernel_size=2, stride=2) -> Output: (B, 32, 14, 14)
    Layer 2:
        - Conv2d(32, 64, kernel_size=3, stride=1, padding=1) -> Output: (B, 64, 14, 14)
        - ReLU() -> Output: (B, 64, 14, 14)
        - MaxPool2d(kernel_size=2, stride=2) -> Output: (B, 64, 7, 7)
    Pooling / Flattening:
        - AdaptiveAvgPool2d((4, 4)) -> Output: (B, 64, 4, 4)
        - Flatten() -> Output: (B, 64 * 4 * 4) = (B, 1024)
    Classifier:
        - Linear(1024, 128) -> Output: (B, 128)
        - ReLU() -> Output: (B, 128)
        - Linear(128, num_classes) -> Output: (B, num_classes) [For MNIST: (B, 10)]
"""

import torch
import torch.nn as nn
from models.base import BaseCNN


class BasicCNN(BaseCNN):
    """Basic Convolutional Neural Network consisting of:
    - 2 Convolutional layers with ReLU activations and Max Pooling
    - Adaptive Average Pooling to fix feature map dimensions
    - 2 Fully Connected layers
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        """Initialize the Basic CNN.

        Args:
            in_channels: Number of input channels (1 for grayscale MNIST, 3 for CIFAR-10).
            num_classes: Number of target classes (default 10).
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Feature Extraction Backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Adaptive pooling ensures consistent flattened dimension across 28x28 and 32x32 inputs
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes)
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits
