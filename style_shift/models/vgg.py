"""VGG-19 encoder for feature extraction in neural style transfer."""

import torch
import torch.nn as nn
from torchvision import models


# VGG-19 layer indices for feature extraction
# Based on torchvision's VGG-19 architecture
VGG_LAYERS = {
    'content': 23,  # conv4_2
    'style': [0, 5, 10, 19, 28],  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
}

# Layer names for better readability
LAYER_NAMES = {
    0: 'conv1_1',
    5: 'conv2_1',
    10: 'conv3_1',
    19: 'conv4_1',
    23: 'conv4_2',
    28: 'conv5_1',
}


def get_vgg19(pretrained: bool = True) -> nn.Sequential:
    """Get VGG-19 model from torchvision.
    
    Args:
        pretrained: Whether to load ImageNet pretrained weights. Default: True
    
    Returns:
        nn.Sequential: VGG-19 feature extractor (classifier head removed)
    
    Examples:
        >>> vgg = get_vgg19(pretrained=True)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = vgg(x)
        >>> features.shape
        torch.Size([1, 512, 7, 7])
    """
    # Load pretrained VGG-19
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
    
    # Get features module (remove classifier head)
    return vgg19.features


class VGG19Encoder(nn.Module):
    """VGG-19 encoder for extracting content and style features.
    
    This encoder uses a pretrained VGG-19 network to extract features from
    content and style images. Content features are extracted from conv4_2,
    and style features are extracted from conv1_1, conv2_1, conv3_1, conv4_1,
    and conv5_1 layers.
    
    The encoder parameters are frozen by default to preserve pretrained weights.
    
    Attributes:
        vgg: VGG-19 feature extractor
        content_layer: Index of content feature layer (conv4_2)
        style_layers: List of indices for style feature layers
    
    Examples:
        >>> encoder = VGG19Encoder()
        >>> content_img = torch.randn(1, 3, 224, 224)
        >>> style_img = torch.randn(1, 3, 224, 224)
        >>> 
        >>> # Extract features
        >>> content_feat, style_feats = encoder(content_img)
        >>> print(content_feat.shape)  # (1, 512, 14, 14)
        >>> print({k: v.shape for k, v in style_feats.items()})
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load VGG-19
        self.vgg = get_vgg19(pretrained=pretrained)
        
        # Store layer indices
        self.content_layer = VGG_LAYERS['content']
        self.style_layers = VGG_LAYERS['style']
        
        # Freeze parameters
        self._freeze_parameters()
    
    def _freeze_parameters(self):
        """Freeze all parameters to preserve pretrained weights."""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Extract content and style features from input image.
        
        Args:
            x: Input image tensor of shape (N, 3, H, W)
        
        Returns:
            tuple:
                - content_feat: Content features from conv4_2, shape (N, 512, H/16, W/16)
                - style_feats: Dictionary of style features from multiple layers
        
        Examples:
            >>> encoder = VGG19Encoder()
            >>> x = torch.randn(1, 3, 224, 224)
            >>> content_feat, style_feats = encoder(x)
        """
        # Initialize output containers
        content_feat = None
        style_feats = {}
        
        # Forward pass through VGG
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            
            # Extract content features
            if i == self.content_layer:
                content_feat = x
            
            # Extract style features
            if i in self.style_layers:
                layer_name = LAYER_NAMES.get(i, f'layer_{i}')
                style_feats[layer_name] = x
        
        return content_feat, style_feats
