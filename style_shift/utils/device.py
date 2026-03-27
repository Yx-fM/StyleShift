"""Device management utilities for CPU/GPU detection and device migration."""

import torch
import torch.nn as nn
from typing import Optional, Union


def is_cuda_available() -> bool:
    """Check if CUDA is available.
    
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Get the best available device.
    
    Args:
        preferred: Optional preferred device type ('cuda', 'cpu', or 'mps').
                   If None, automatically selects the best available device.
    
    Returns:
        torch.device: The selected device.
    
    Examples:
        >>> device = get_device()
        >>> print(device)
        device(type='cuda')  # or device(type='cpu')
        
        >>> device = get_device('cpu')
        >>> print(device)
        device(type='cpu')
    """
    if preferred is not None:
        preferred = preferred.lower()
        if preferred == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("Warning: CUDA not available, falling back to CPU")
                return torch.device('cpu')
        elif preferred == 'mps':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                print("Warning: MPS not available, falling back to CPU")
                return torch.device('cpu')
        elif preferred == 'cpu':
            return torch.device('cpu')
        else:
            raise ValueError(f"Unknown device type: {preferred}. Must be 'cuda', 'cpu', or 'mps'.")
    
    # Automatic selection
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def to_device(
    model: nn.Module,
    device: Optional[Union[torch.device, str]] = None
) -> nn.Module:
    """Move a model to the specified device.
    
    Args:
        model: PyTorch model to move.
        device: Target device. If None, automatically selects the best device.
    
    Returns:
        nn.Module: The model on the target device.
    
    Examples:
        >>> from torchvision.models import resnet18
        >>> model = resnet18()
        >>> model = to_device(model)  # Auto-detect
        >>> model = to_device(model, 'cuda')  # Force CUDA
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)
    
    return model.to(device)


def get_device_name(device: Optional[torch.device] = None) -> str:
    """Get human-readable device name.
    
    Args:
        device: torch.device to get name for. If None, uses current device.
    
    Returns:
        str: Human-readable device name.
    """
    if device is None:
        device = get_device()
    
    if device.type == 'cuda':
        return f"CUDA {torch.cuda.get_device_name(device.index if device.index else 0)}"
    elif device.type == 'mps':
        return "Apple Silicon (MPS)"
    else:
        return "CPU"
