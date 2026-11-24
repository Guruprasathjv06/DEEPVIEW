"""Device detection and automatic hardware acceleration selection for DEEPVISION."""

import logging
from typing import Optional
import torch

logger = logging.getLogger(__name__)


def get_device(requested_device: Optional[str] = "auto") -> torch.device:
    """Detect available hardware acceleration and return the appropriate torch.device.

    Args:
        requested_device: Device preference: 'auto', 'cuda', or 'cpu'.
                         Defaults to 'auto'.

    Returns:
        torch.device: Selected PyTorch device (CUDA or CPU).
    """
    if requested_device is None:
        requested_device = "auto"

    requested_device = requested_device.lower().strip()

    if requested_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Explicitly selected CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    if requested_device == "cpu":
        logger.info("Explicitly selected CPU device.")
        return torch.device("cpu")

    # Automatic detection ('auto')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"[Hardware Detection] NVIDIA CUDA is AVAILABLE: {gpu_name} ({vram_gb:.2f} GB VRAM, {gpu_count} device(s)).")
        logger.info("Automatically selected device: cuda:0")
        return torch.device("cuda:0")
    else:
        logger.info("[Hardware Detection] NVIDIA CUDA is NOT available on this system.")
        logger.info("Automatically selected device: cpu")
        return torch.device("cpu")


def get_device_info(device: torch.device) -> dict:
    """Return hardware diagnostic details as a dictionary.

    Args:
        device: The active torch.device.

    Returns:
        dict: Diagnostic details about the runtime device.
    """
    info = {
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["device_count"] = torch.cuda.device_count()
        info["vram_total_gb"] = round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2)
    else:
        info["gpu_name"] = "N/A (CPU execution)"
        info["device_count"] = 0
        info["vram_total_gb"] = 0.0

    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dev = get_device("auto")
    print("Selected Device:", dev)
    print("Device Info:", get_device_info(dev))
