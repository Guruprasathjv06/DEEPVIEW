"""Experiment configuration classes for CIFAR-10 in DEEPVISION."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class CIFAR10Config:
    """Hyperparameters and configuration for CIFAR-10 training & cross-validation."""

    # Project metadata
    project_name: str = "DEEPVISION"
    experiment_name: str = "cifar10_experiment"
    model_name: str = "improved"  # 'basic' or 'improved'
    dataset_name: str = "CIFAR-10"

    # Hardware & reproducibility
    device: str = "auto"
    seed: int = 42

    # Cross-validation
    n_splits: int = 5
    augment_train: bool = True

    # Training hyperparameters
    epochs: int = 5
    batch_size: int = 64
    optimizer_name: str = "adam"  # 'adam' or 'sgd'
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    lr_step_size: int = 3
    lr_gamma: float = 0.5

    # Checkpoints & outputs
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    save_best_only: bool = True

    # Weights & Biases
    use_wandb: bool = True
    wandb_mode: str = "online"  # 'online', 'offline', or 'disabled'
    wandb_entity: Optional[str] = None
    wandb_tags: list = field(default_factory=lambda: ["cifar10", "cross-validation"])
    log_checkpoint_artifacts: bool = False

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, filepath: str) -> None:
        """Save configuration as JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> "CIFAR10Config":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)
