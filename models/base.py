"""Base neural network class with model utilities for DEEPVISION."""

import torch
import torch.nn as nn
from typing import Dict, Any


class BaseCNN(nn.Module):
    """Abstract base class for CNN models providing parameter inspection and summaries."""

    def __init__(self) -> None:
        super().__init__()

    def count_parameters(self) -> Dict[str, int]:
        """Count total, trainable, and non-trainable parameters.

        Returns:
            Dict[str, int]: Parameter statistics.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
        }

    def get_model_summary(self, input_size: tuple = (1, 1, 28, 28)) -> Dict[str, Any]:
        """Generate a summary of the model structure and tensor dimensions.

        Args:
            input_size: Dummy input tensor shape (B, C, H, W).

        Returns:
            Dict[str, Any]: Model summary information.
        """
        params = self.count_parameters()
        device = next(self.parameters()).device
        dummy_input = torch.zeros(input_size, device=device)
        self.eval()
        with torch.no_grad():
            output = self(dummy_input)

        return {
            "model_name": self.__class__.__name__,
            "input_shape": list(input_size),
            "output_shape": list(output.shape),
            **params,
        }
