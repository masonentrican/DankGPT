"""
Checkpoint utilities for saving and loading model states.

Provides centralized checkpoint management for models, optimizers, and training metadata.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any


def save_checkpoint(
    model: torch.nn.Module,
    filepath: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    **kwargs: Any
) -> None:
    """
    Save a model checkpoint with optional optimizer state and metadata.
    
    Args:
        model: The model to save.
        filepath: Path where to save the checkpoint.
        optimizer: Optional optimizer to save state for.
        epoch: Optional epoch number.
        loss: Optional loss value.
        **kwargs: Additional metadata to save in the checkpoint.
    
    Example:
        >>> save_checkpoint(model, Path("checkpoint.pth"), optimizer, epoch=10, loss=2.5)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if loss is not None:
        checkpoint["loss"] = loss
    
    # Add any additional metadata
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load a model checkpoint with optional optimizer state.
    
    Args:
        filepath: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        device: Device to map the checkpoint to. If None, uses model's device.
        strict: Whether to strictly enforce that the keys in state_dict match.
    
    Returns:
        Dictionary containing checkpoint metadata (epoch, loss, etc.) if present.
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    
    Example:
        >>> metadata = load_checkpoint(Path("checkpoint.pth"), model, optimizer, device)
        >>> epoch = metadata.get("epoch", 0)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
    
    # Determine device for map_location
    if device is None:
        device = next(model.parameters()).device
    
    map_location = str(device) if device.type != "cpu" else "cpu"
    
    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=map_location)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Return metadata (everything except model and optimizer state)
    metadata = {k: v for k, v in checkpoint.items() 
                if k not in ["model_state_dict", "optimizer_state_dict"]}
    
    return metadata

