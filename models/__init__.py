"""Model architectures for DEEPVISION."""

from typing import Union
from models.base import BaseCNN
from models.basic_cnn import BasicCNN
from models.improved_cnn import ImprovedCNN

__all__ = ["BaseCNN", "BasicCNN", "ImprovedCNN", "get_model"]


def get_model(
    model_name: str,
    in_channels: int = 1,
    num_classes: int = 10,
    **kwargs,
) -> BaseCNN:
    """Factory function to instantiate CNN models by name.

    Args:
        model_name: Name of architecture ('basic', 'improved', 'basic_cnn', 'improved_cnn').
        in_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10).
        num_classes: Number of target output classes.
        **kwargs: Additional parameters passed to model constructor.

    Returns:
        BaseCNN: Instantiated PyTorch model.
    """
    normalized_name = model_name.lower().replace("-", "_").replace(" ", "_")

    if normalized_name in ("basic", "basic_cnn"):
        return BasicCNN(in_channels=in_channels, num_classes=num_classes)
    elif normalized_name in ("improved", "improved_cnn"):
        return ImprovedCNN(in_channels=in_channels, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. Available architectures: 'basic', 'improved'."
        )
