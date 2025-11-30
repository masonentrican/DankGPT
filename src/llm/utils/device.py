"""
Device utilities for PyTorch operations.

Provides centralized device selection to enable
consistent device handling across the codebase.
"""

import torch
from torch import device as Device


def get_device(device_str: str = "auto") -> Device:
    """
    Get the appropriate PyTorch device.
    
    Automatically selects the best available device or uses the specified one.
    Supports CUDA, MPS (Apple Silicon), and CPU.
    
    Args:
        device_str: Device specification. Options:
            - "auto": Automatically select best available (cuda > mps > cpu)
            - "cuda": Use CUDA if available, otherwise CPU
            - "mps": Use MPS (Apple) if available, otherwise CPU
            - "cpu": Force CPU
            - Any valid torch.device string
    
    Returns:
        torch.device: The selected device.
    
    Example:
        >>> device = get_device("auto")
        >>> model.to(device)
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    elif device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    elif device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    elif device_str == "cpu":
        return torch.device("cpu")
    
    else:
        # Allow any valid torch.device string to pass through
        return torch.device(device_str)

