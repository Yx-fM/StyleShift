"""Decoder network for neural style transfer."""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Decoder network that reconstructs image from AdaIN features.
    
    The decoder takes style-transferred features and reconstructs them
    back into image space. It uses a series of convolutional layers with
    upsampling to progressively increase spatial resolution.
    
    Architecture:
        Input: (N, 512, H/16, W/16)
        → Conv + ReLU + Upsample × 4
        → Output: (N, 3, H, W)
    
    Attributes:
        layers: Sequential module containing decoder layers
    
    Examples:
        >>> decoder = Decoder()
        >>> features = torch.randn(1, 512, 14, 14)
        >>> image = decoder(features)
        >>> image.shape
        torch.Size([1, 3, 224, 224])
    """
    
    def __init__(self):
        super().__init__()
        
        # Decoder architecture
        self.layers = nn.Sequential(
            # Block 1: 512 -> 256
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 2: 256 -> 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 3: 256 -> 128
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 4: 128 -> 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Output: 64 -> 3 (RGB)
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode features back to image space.
        
        Args:
            x: Feature tensor from AdaIN, shape (N, 512, H/16, W/16)
        
        Returns:
            torch.Tensor: Reconstructed image, shape (N, 3, H, W)
            Values are in range [0, 1] after sigmoid activation.
        
        Examples:
            >>> decoder = Decoder()
            >>> x = torch.randn(2, 512, 14, 14)
            >>> output = decoder(x)
            >>> output.shape
            torch.Size([2, 3, 224, 224])
        """
        return self.layers(x)
