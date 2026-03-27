"""Main style transfer orchestrator using AdaIN."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional, Dict, List
from PIL import Image
from dataclasses import dataclass

from ..models.vgg import VGG19Encoder
from ..models.adain import AdaIN, adain_function
from ..models.decoder import Decoder
from .preprocess import (
    prepare_content_tensor,
    prepare_style_tensor,
    preprocess_image,
    blend_images,
    preserve_original_colors,
)
from .postprocess import (
    postprocess_image,
    save_result,
)


@dataclass
class StyleTransferConfig:
    """Configuration for style transfer."""
    alpha: float = 1.0          # Style strength (0.0-1.0)
    content_weight: float = 1.0
    style_weight: float = 1.0
    tv_weight: float = 0.0001   # Total variation regularization
    max_size: int = 512
    device: Optional[str] = None
    preserve_color: bool = False
    crop: Optional[str] = None  # 'center' | 'random' | None


class StyleTransfer:
    """Main style transfer orchestrator using AdaIN.
    
    This class provides a high-level interface for performing neural style transfer
    using the AdaIN (Adaptive Instance Normalization) algorithm. It handles all
    preprocessing, feature extraction, style fusion, and decoding automatically.
    
    Args:
        config: Configuration object. If None, uses default config.
        vgg: Pre-loaded VGG encoder. If None, loads automatically.
        decoder: Pre-loaded decoder. If None, loads automatically.
    
    Attributes:
        config: Style transfer configuration
        vgg: VGG-19 encoder for feature extraction
        adain: AdaIN layer for style fusion
        decoder: Decoder network for image reconstruction
    
    Examples:
        >>> st = StyleTransfer()
        >>> result = st.transfer(
        ...     content='photo.jpg',
        ...     style='anime.jpg',
        ...     alpha=0.8
        ... )
        >>> result.save('output.jpg')
    """
    
    # Built-in style names and their file paths
    BUILTIN_STYLES = {
        'anime': 'styles/anime.jpg',
        'vangogh': 'styles/vangogh.jpg',
        'monet': 'styles/monet.jpg',
        'ukiyoe': 'styles/ukiyoe.jpg',
        'mosaic': 'styles/mosaic.jpg',
        'sketch': 'styles/sketch.jpg',
        'watercolor': 'styles/watercolor.jpg',
    }
    
    def __init__(
        self,
        config: Optional[StyleTransferConfig] = None,
        vgg: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None
    ):
        """Initialize StyleTransfer with models and configuration."""
        self.config = config or StyleTransferConfig()
        
        # Determine device
        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            # Auto-detect best device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        
        # Load models
        self.vgg = vgg if vgg is not None else VGG19Encoder(pretrained=True)
        self.vgg.to(self.device)
        self.vgg.eval()
        
        self.decoder = decoder if decoder is not None else Decoder()
        self.decoder.to(self.device)
        self.decoder.eval()
        
        self.adain = AdaIN()
    
    def transfer(
        self,
        content: Union[str, Path, Image.Image, torch.Tensor],
        style: Optional[Union[str, Path, Image.Image, torch.Tensor]] = None,
        style_name: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        alpha: Optional[float] = None
    ) -> Union[Image.Image, torch.Tensor]:
        """Perform style transfer on content image.
        
        Args:
            content: Content image (path, PIL, or tensor)
            style: Style image (optional if style_name provided)
            style_name: Built-in style name (anime, vangogh, monet, etc.)
            output_path: Save location (optional)
            alpha: Override style strength (0.0-1.0)
        
        Returns:
            Stylized image (PIL if content was PIL/path, tensor if tensor)
        
        Raises:
            ValueError: If neither style nor style_name provided
            FileNotFoundError: If style image not found
        """
        # Validate inputs
        if style is None and style_name is None:
            raise ValueError("Must provide either 'style' or 'style_name'")
        
        # Use alpha from parameter or config
        alpha = alpha if alpha is not None else self.config.alpha
        
        # Prepare content tensor
        content_tensor = prepare_content_tensor(
            content,
            max_size=self.config.max_size
        ).to(self.device)
        
        # Prepare style tensor
        if style_name is not None:
            # Load built-in style
            style_tensor = self._load_builtin_style(style_name)
        else:
            style_tensor = prepare_style_tensor(
                style,
                max_size=self.config.max_size
            ).to(self.device)
        
        # Perform style transfer
        with torch.no_grad():
            # Extract features
            content_feat = self._encode_content(content_tensor)
            style_feats = self._encode_style(style_tensor)
            
            # Apply AdaIN
            stylized_feat = self._apply_adain(content_feat, style_feats)
            
            # Decode to image
            output_tensor = self._decode(stylized_feat)
            
            # Preserve colors if requested
            if self.config.preserve_color:
                output_tensor = preserve_original_colors(
                    output_tensor,
                    content_tensor,
                    preserve_ratio=alpha * 0.3  # Subtle color preservation
                )
        
        # Postprocess
        output_image = postprocess_image(output_tensor, denormalize_flag=True)
        
        # Save if requested
        if output_path:
            save_result(output_tensor, output_path)
        
        return output_image
    
    def transfer_batch(
        self,
        contents: List[Union[str, Path, Image.Image, torch.Tensor]],
        style: Union[str, Path, Image.Image, torch.Tensor],
        output_paths: Optional[List[Union[str, Path]]] = None,
        alpha: Optional[float] = None
    ) -> List[Union[Image.Image, torch.Tensor]]:
        """Process multiple content images with same style.
        
        Args:
            contents: List of content images
            style: Style image (single)
            output_paths: Optional list of output paths
            alpha: Style strength
        
        Returns:
            List of stylized images
        """
        results = []
        
        for i, content in enumerate(contents):
            output_path = output_paths[i] if output_paths else None
            result = self.transfer(
                content=content,
                style=style,
                output_path=output_path,
                alpha=alpha
            )
            results.append(result)
        
        return results
    
    def style_interpolation(
        self,
        content: Union[str, Path, Image.Image, torch.Tensor],
        style: Union[str, Path, Image.Image, torch.Tensor],
        alpha: float = 0.5
    ) -> Union[Image.Image, torch.Tensor]:
        """Interpolate between content and style.
        
        Args:
            content: Content image
            style: Style image
            alpha: Interpolation factor (0.0 = content, 1.0 = full style transfer)
        
        Returns:
            Interpolated stylized image
        """
        if alpha <= 0.0:
            # Return content unchanged
            if isinstance(content, Image.Image):
                return content
            elif isinstance(content, (str, Path)):
                return Image.open(content)
            else:
                return postprocess_image(content)
        
        if alpha >= 1.0:
            # Full style transfer
            return self.transfer(content=content, style=style, alpha=1.0)
        
        # Interpolate
        result = self.transfer(content=content, style=style, alpha=alpha)
        return result
    
    @classmethod
    def get_builtin_styles(cls) -> Dict[str, str]:
        """Return dict of built-in style names to descriptions.
        
        Returns:
            Dict mapping style names to descriptions
        """
        return {
            'anime': '二次元风格（动漫化）',
            'vangogh': '梵高《星夜》风格',
            'monet': '莫奈印象派风格',
            'ukiyoe': '浮世绘风格',
            'mosaic': '马赛克风格',
            'sketch': '铅笔素描',
            'watercolor': '水彩画风格',
        }
    
    def _encode_content(self, content_tensor: torch.Tensor) -> torch.Tensor:
        """Extract content features from VGG (conv4_2).
        
        Args:
            content_tensor: Content image tensor (1, C, H, W)
        
        Returns:
            Content feature tensor (1, 512, H/8, W/8)
        """
        _, style_feats = self.vgg(content_tensor)
        # Use conv4_2 as content feature
        return style_feats.get('conv4_2', None) or list(style_feats.values())[-1]
    
    def _encode_style(self, style_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract style features from VGG (conv1_1, conv2_1, ..., conv5_1).
        
        Args:
            style_tensor: Style image tensor (1, C, H, W)
        
        Returns:
            Dictionary of style features at multiple layers
        """
        _, style_feats = self.vgg(style_tensor)
        return style_feats
    
    def _apply_adain(
        self,
        content_feat: torch.Tensor,
        style_feats: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply AdaIN to merge content and style features.
        
        Args:
            content_feat: Content feature tensor
            style_feats: Dictionary of style feature tensors
        
        Returns:
            Stylized feature tensor
        """
        # Use conv4_2 for style statistics as well (matching content feature size)
        style_feat = style_feats.get('conv4_2', None)
        if style_feat is None:
            # Fallback to closest layer
            style_feat = list(style_feats.values())[-1]
        
        return adain_function(content_feat, style_feat)
    
    def _decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to image space via decoder network.
        
        Args:
            features: Stylized feature tensor (1, 512, H/8, W/8)
        
        Returns:
            Output image tensor (1, 3, H, W)
        """
        return self.decoder(features)
    
    def _load_builtin_style(self, style_name: str) -> torch.Tensor:
        """Load built-in style image tensor.
        
        Args:
            style_name: Name of built-in style
        
        Returns:
            Style image tensor (1, C, H, W)
        
        Raises:
            ValueError: If style_name not found
        """
        style_name_lower = style_name.lower()
        
        if style_name_lower not in self.BUILTIN_STYLES:
            available = ', '.join(self.BUILTIN_STYLES.keys())
            raise ValueError(
                f"Unknown style '{style_name}'. Available styles: {available}"
            )
        
        style_path = Path(self.BUILTIN_STYLES[style_name_lower])
        
        # Check if file exists
        if not style_path.exists():
            # Return placeholder (white image) if style not found
            print(f"Warning: Style file not found: {style_path}")
            return torch.ones(1, 3, 256, 256, device=self.device) * 0.5
        
        # Load and preprocess style
        style_tensor = prepare_style_tensor(
            style_path,
            max_size=self.config.max_size
        ).to(self.device)
        
        return style_tensor
