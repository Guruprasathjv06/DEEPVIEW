"""Training engine, device detection, and callbacks for DEEPVISION."""

from training.device import get_device, get_device_info
from training.early_stopping import EarlyStopping
from training.trainer import ModelTrainer

__all__ = [
    "get_device",
    "get_device_info",
    "EarlyStopping",
    "ModelTrainer",
]
