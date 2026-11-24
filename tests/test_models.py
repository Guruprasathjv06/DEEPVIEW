"""Unit tests for CNN models in DEEPVISION."""

import pytest
import torch
from models import BasicCNN, ImprovedCNN, get_model


class TestCNNModels:
    """Test suite for BasicCNN and ImprovedCNN architectures."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_basic_cnn_mnist_output_shape(self, batch_size: int):
        """Verify BasicCNN produces (B, 10) logits for MNIST-sized inputs (1x28x28)."""
        model = BasicCNN(in_channels=1, num_classes=10)
        x = torch.randn(batch_size, 1, 28, 28)
        out = model(x)
        assert out.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {out.shape}"
        assert not torch.isnan(out).any(), "Model output contains NaNs"

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_improved_cnn_mnist_output_shape(self, batch_size: int):
        """Verify ImprovedCNN produces (B, 10) logits for MNIST-sized inputs (1x28x28)."""
        model = ImprovedCNN(in_channels=1, num_classes=10)
        x = torch.randn(batch_size, 1, 28, 28)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {out.shape}"
        assert not torch.isnan(out).any(), "Model output contains NaNs"

    def test_model_parameter_counts(self):
        """Verify parameter counting on models."""
        basic = BasicCNN(in_channels=1, num_classes=10)
        improved = ImprovedCNN(in_channels=1, num_classes=10)

        basic_params = basic.count_parameters()
        improved_params = improved.count_parameters()

        assert basic_params["total_parameters"] > 0
        assert basic_params["trainable_parameters"] == basic_params["total_parameters"]

        assert improved_params["total_parameters"] > 0
        assert improved_params["trainable_parameters"] == improved_params["total_parameters"]

        # Improved CNN should have more parameters due to deeper layers and batch normalization
        assert improved_params["total_parameters"] > basic_params["total_parameters"]

    def test_factory_function(self):
        """Test model creation via get_model factory."""
        basic = get_model("basic", in_channels=1, num_classes=10)
        assert isinstance(basic, BasicCNN)

        improved = get_model("improved", in_channels=1, num_classes=10)
        assert isinstance(improved, ImprovedCNN)

        with pytest.raises(ValueError):
            get_model("invalid_model_name")

    def test_cifar_compatibility_shape(self):
        """Verify models can accept 3-channel 32x32 inputs (for CIFAR-10 readiness)."""
        x = torch.randn(2, 3, 32, 32)
        basic = BasicCNN(in_channels=3, num_classes=10)
        improved = ImprovedCNN(in_channels=3, num_classes=10)

        out_b = basic(x)
        improved.eval()
        with torch.no_grad():
            out_i = improved(x)

        assert out_b.shape == (2, 10)
        assert out_i.shape == (2, 10)
