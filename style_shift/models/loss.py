"""Loss functions for neural style transfer training."""

import torch
import torch.nn as nn
from typing import Dict


class ContentLoss(nn.Module):
    """Content loss using MSE between feature representations.
    
    Computes Mean Squared Error between content and generated image features.
    Lower loss means the generated image has similar content structure.
    
    Args:
        weight: Loss weight. Default: 1.0
    
    Examples:
        >>> loss_fn = ContentLoss(weight=1.0)
        >>> content_feat = torch.randn(1, 512, 14, 14)
        >>> gen_feat = torch.randn(1, 512, 14, 14)
        >>> loss = loss_fn(gen_feat, content_feat)
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()
    
    def forward(self, generated: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """Compute content loss.
        
        Args:
            generated: Generated image features
            content: Content image features
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        return self.weight * self.mse(generated, content)


class StyleLoss(nn.Module):
    """Style loss using Gram matrices.
    
    Computes MSE between Gram matrices of style and generated image features.
    Lower loss means the generated image has similar texture/color statistics.
    
    The Gram matrix captures correlations between filter responses,
    representing the texture and style of an image.
    
    Args:
        weight: Loss weight. Default: 1.0
    
    Examples:
        >>> loss_fn = StyleLoss(weight=1.0)
        >>> style_feat = torch.randn(1, 512, 28, 28)
        >>> gen_feat = torch.randn(1, 512, 28, 28)
        >>> loss = loss_fn(gen_feat, style_feat)
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.mse = nn.MSELoss()
    
    def _gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix of features.
        
        Args:
            x: Feature tensor of shape (N, C, H, W)
        
        Returns:
            torch.Tensor: Gram matrix of shape (N, C, C)
        """
        N, C, H, W = x.shape
        
        # Normalize features
        x_normalized = x / (C * H * W)
        
        # Reshape to (N, C, H*W)
        x_reshaped = x_normalized.view(N, C, H * W)
        
        # Compute Gram matrix: (N, C, H*W) @ (N, H*W, C) = (N, C, C)
        gram = torch.bmm(x_reshaped, x_reshaped.transpose(1, 2))
        
        return gram
    
    def forward(self, generated: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Compute style loss.
        
        Args:
            generated: Generated image features
            style: Style image features
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        gram_gen = self._gram_matrix(generated)
        gram_style = self._gram_matrix(style)
        
        return self.weight * self.mse(gram_gen, gram_style)


class TVLoss(nn.Module):
    """Total Variation loss for spatial smoothing.
    
    Penalizes high-frequency noise in generated images by encouraging
    neighboring pixels to have similar values.
    
    Formula:
        TV = sum(|x[i,j] - x[i+1,j]| + |x[i,j] - x[i,j+1]|)
    
    Args:
        weight: Loss weight. Default: 1e-6
    
    Examples:
        >>> loss_fn = TVLoss(weight=1e-6)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> loss = loss_fn(image)
    """
    
    def __init__(self, weight: float = 1e-6):
        super().__init__()
        self.weight = weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss.
        
        Args:
            x: Image tensor of shape (N, C, H, W)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Horizontal differences
        diff_h = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        
        # Vertical differences
        diff_v = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        
        # Sum and average
        tv = diff_h.sum() + diff_v.sum()
        
        return self.weight * tv


def combine_losses(
    content_loss: torch.Tensor,
    style_loss: torch.Tensor,
    tv_loss: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0
) -> torch.Tensor:
    """Combine multiple loss components.
    
    Args:
        content_loss: Content loss value
        style_loss: Style loss value
        tv_loss: Total variation loss (optional)
        alpha: Content loss weight. Default: 1.0
        beta: Style loss weight. Default: 1.0
        gamma: TV loss weight. Default: 0.0
    
    Returns:
        torch.Tensor: Combined loss value
    
    Examples:
        >>> c_loss = torch.tensor(1.0)
        >>> s_loss = torch.tensor(2.0)
        >>> total = combine_losses(c_loss, s_loss, alpha=1.0, beta=0.1)
    """
    total = alpha * content_loss + beta * style_loss
    
    if tv_loss is not None and gamma > 0:
        total = total + gamma * tv_loss
    
    return total
