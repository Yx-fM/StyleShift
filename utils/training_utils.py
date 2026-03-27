"""Training utilities for StyleShift."""

import torch
import torch.nn as nn
from typing import Dict, Tuple


def get_vgg_features(vgg: nn.Module, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Extract VGG features from image.
    
    Args:
        vgg: VGG-19 encoder (frozen)
        image: Input image tensor (N, 3, H, W)
    
    Returns:
        Tuple of (content_feature, style_features_dict)
    """
    with torch.no_grad():
        content_feat, style_feats = vgg(image)
    return content_feat, style_feats


def calc_mean_std(feature: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate mean and std of features.
    
    Args:
        feature: Feature tensor (N, C, H, W)
        eps: Small value for numerical stability
    
    Returns:
        Tuple of (mean, std) tensors
    """
    size = feature.size()
    assert len(size) == 4
    
    # Calculate mean and std over spatial dimensions (H, W)
    mean = feature.mean(dim=[2, 3], keepdim=True)
    std = feature.std(dim=[2, 3], keepdim=True) + eps
    
    return mean, std


def normalize_tensor(tensor: torch.Tensor, 
                     mean: list = [0.485, 0.456, 0.406],
                     std: list = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """Normalize tensor with ImageNet statistics.
    
    Args:
        tensor: Image tensor (N, 3, H, W) in [0, 1]
        mean: Mean values for each channel
        std: Std values for each channel
    
    Returns:
        Normalized tensor
    """
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor,
                       mean: list = [0.485, 0.456, 0.406],
                       std: list = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """Denormalize tensor from ImageNet statistics to [0, 1] range.
    
    Args:
        tensor: Normalized tensor (N, 3, H, W)
        mean: Mean values for each channel
        std: Std values for each channel
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    result = tensor * std + mean
    return result.clamp(0, 1)


def create_image_grid(images: torch.Tensor, nrow: int = 4) -> torch.Tensor:
    """Create a grid of images for visualization.
    
    Args:
        images: Batch of images (N, 3, H, W)
        nrow: Number of images per row
    
    Returns:
        Grid image tensor
    """
    from torchvision.utils import make_grid
    return make_grid(images, nrow=nrow, normalize=True, value_range=(0, 1))


def save_image(tensor: torch.Tensor, path: str):
    """Save tensor as image file.
    
    Args:
        tensor: Image tensor (3, H, W) or (N, 3, H, W)
        path: Output file path
    """
    from torchvision.utils import save_image
    save_image(tensor, path, normalize=True, value_range=(0, 1))
