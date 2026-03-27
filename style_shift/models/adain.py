"""AdaIN (Adaptive Instance Normalization) implementation for style transfer."""

import torch
import torch.nn as nn


def adain_function(content: torch.Tensor, style: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply Adaptive Instance Normalization.
    
    AdaIN aligns the mean and variance of the content features with those of the style features.
    
    Mathematical formulation:
        AdaIN(c, s) = σ(s) * (c - μ(c)) / σ(c) + μ(s)
    
    where:
        - c: content features
        - s: style features
        - μ: mean across spatial dimensions (H, W)
        - σ: standard deviation across spatial dimensions (H, W)
        - eps: small constant for numerical stability
    
    Args:
        content: Content feature tensor of shape (N, C, H, W)
        style: Style feature tensor of shape (N, C, H, W)
        eps: Small constant to prevent division by zero. Default: 1e-5
    
    Returns:
        torch.Tensor: Style-transferred features with shape (N, C, H, W)
    
    Examples:
        >>> content = torch.randn(2, 512, 14, 14)
        >>> style = torch.randn(2, 512, 14, 14)
        >>> output = adain_function(content, style)
        >>> output.shape
        torch.Size([2, 512, 14, 14])
    """
    # Extract dimensions
    N, C, H, W = content.shape
    
    # Calculate mean and variance for content
    content_mean = content.mean(dim=[2, 3], keepdim=True)  # (N, C, 1, 1)
    content_var = content.var(dim=[2, 3], keepdim=True)    # (N, C, 1, 1)
    
    # Calculate mean and variance for style
    style_mean = style.mean(dim=[2, 3], keepdim=True)      # (N, C, 1, 1)
    style_var = style.var(dim=[2, 3], keepdim=True)        # (N, C, 1, 1)
    
    # Normalize content features
    content_normalized = (content - content_mean) / torch.sqrt(content_var + eps)
    
    # Apply style's mean and variance
    output = content_normalized * torch.sqrt(style_var + eps) + style_mean
    
    return output


class AdaIN(nn.Module):
    """AdaIN (Adaptive Instance Normalization) layer.
    
    This module wraps the adain_function as a PyTorch nn.Module,
    making it easy to integrate into neural network architectures.
    
    Args:
        eps: Small constant to prevent division by zero. Default: 1e-5
    
    Attributes:
        eps: Numerical stability constant
    
    Examples:
        >>> adain = AdaIN()
        >>> content = torch.randn(2, 512, 14, 14)
        >>> style = torch.randn(2, 512, 14, 14)
        >>> output = adain(content, style)
        >>> output.shape
        torch.Size([2, 512, 14, 14])
    """
    
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply AdaIN to content features using style statistics.
        
        Args:
            content: Content feature tensor of shape (N, C, H, W)
            style: Style feature tensor of shape (N, C, H, W)
        
        Returns:
            torch.Tensor: Style-transferred features with shape (N, C, H, W)
        """
        return adain_function(content, style, self.eps)
