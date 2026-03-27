"""Image postprocessing utilities for style transfer."""

import torch
from PIL import Image
from pathlib import Path
from typing import Union, Optional
from torchvision import transforms

from .preprocess import denormalize, IMAGENET_MEAN, IMAGENET_STD


def clamp_tensor(
    tensor: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> torch.Tensor:
    """Clamp tensor values to valid image range.
    
    Args:
        tensor: Image tensor to clamp
        min_val: Minimum value. Default: 0.0
        max_val: Maximum value. Default: 1.0
    
    Returns:
        torch.Tensor: Clamped tensor
    """
    return tensor.clamp(min_val, max_val)


def postprocess_image(
    tensor: torch.Tensor,
    denormalize_flag: bool = True,
    format: str = 'RGB'
) -> Image.Image:
    """Convert processed tensor to PIL Image.
    
    Args:
        tensor: Image tensor in shape (C, H, W) or (1, C, H, W)
        denormalize_flag: Whether to denormalize from ImageNet stats. Default: True
        format: Output image format (RGB, L, etc.). Default: 'RGB'
    
    Returns:
        Image.Image: PIL Image ready for display or saving
    
    Raises:
        ValueError: If tensor has invalid shape
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        else:
            raise ValueError(f"Expected batch size 1, got {tensor.shape[0]}")
    
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor (C, H, W), got {tensor.dim()}D")
    
    # Denormalize if needed
    if denormalize_flag:
        tensor = denormalize(tensor)
    
    # Clamp to valid range
    tensor = clamp_tensor(tensor, 0.0, 1.0)
    
    # Convert to PIL Image
    transform = transforms.ToPILImage()
    image = transform(tensor)
    
    # Convert to specified format
    if format and image.mode != format:
        image = image.convert(format)
    
    return image


def save_result(
    tensor: torch.Tensor,
    path: Union[str, Path],
    format: Optional[str] = None,
    quality: int = 95
) -> None:
    """Save result tensor to image file (auto-detect format).
    
    Args:
        tensor: Image tensor in shape (C, H, W) or (1, C, H, W)
        path: Output file path
        format: Image format (JPEG, PNG, etc.). If None, auto-detected from extension
        quality: JPEG quality (1-100). Default: 95
    
    Raises:
        ValueError: If tensor has invalid shape
    """
    path = Path(path)
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect format from extension
    if format is None:
        ext = path.suffix.lower()
        format_map = {
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.png': 'PNG',
            '.bmp': 'BMP',
            '.webp': 'WEBP',
            '.tiff': 'TIFF',
        }
        format = format_map.get(ext, 'PNG')  # Default to PNG
    
    # Postprocess and save
    image = postprocess_image(tensor, denormalize_flag=True, format='RGB')
    
    # Save with appropriate parameters
    save_kwargs = {}
    if format == 'JPEG':
        save_kwargs['quality'] = quality
        save_kwargs['optimize'] = True
    elif format == 'PNG':
        save_kwargs['compress_level'] = 6
    elif format == 'WEBP':
        save_kwargs['quality'] = quality
    
    image.save(path, format=format, **save_kwargs)


def blend_images(
    image1: torch.Tensor,
    image2: torch.Tensor,
    alpha: float = 0.5
) -> torch.Tensor:
    """Linear interpolation between two images.
    
    Args:
        image1: First image tensor
        image2: Second image tensor
        alpha: Blend factor (0.0 = image1, 1.0 = image2)
    
    Returns:
        torch.Tensor: Blended image tensor
    """
    alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
    return (1 - alpha) * image1 + alpha * image2


def preserve_original_colors(
    stylized: torch.Tensor,
    original: torch.Tensor,
    preserve_ratio: float = 0.5
) -> torch.Tensor:
    """Preserve original image colors in stylized result.
    
    Args:
        stylized: Stylized image tensor
        original: Original content image tensor
        preserve_ratio: How much to preserve original colors (0.0-1.0)
    
    Returns:
        torch.Tensor: Stylized image with preserved colors
    """
    return blend_images(stylized, original, preserve_ratio)


def create_comparison(
    original: torch.Tensor,
    stylized: torch.Tensor,
    side_by_side: bool = True
) -> Image.Image:
    """Create comparison image of original and stylized.
    
    Args:
        original: Original image tensor
        stylized: Stylized image tensor
        side_by_side: If True, place images side by side; if False, stack vertically
    
    Returns:
        Image.Image: Comparison image
    """
    # Ensure both tensors are on same device and have same shape
    if original.shape != stylized.shape:
        # Resize stylized to match original
        from torchvision import transforms
        resize = transforms.Resize((original.shape[1], original.shape[2]))
        stylized = resize(stylized)
    
    # Convert to PIL
    orig_pil = postprocess_image(original)
    styl_pil = postprocess_image(stylized)
    
    # Create comparison
    if side_by_side:
        width, height = orig_pil.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(orig_pil, (0, 0))
        comparison.paste(styl_pil, (width, 0))
    else:
        width, height = orig_pil.size
        comparison = Image.new('RGB', (width, height * 2))
        comparison.paste(orig_pil, (0, 0))
        comparison.paste(styl_pil, (0, height))
    
    return comparison


def add_label(
    image: Image.Image,
    label: str,
    position: str = 'top'
) -> Image.Image:
    """Add text label to image.
    
    Args:
        image: PIL Image
        label: Text label
        position: Label position ('top' or 'bottom')
    
    Returns:
        Image.Image: Labeled image
    """
    from PIL import ImageDraw, ImageFont
    
    # Create a new image with space for label
    width, height = image.size
    label_height = 30
    total_height = height + label_height
    
    if position == 'top':
        labeled = Image.new('RGB', (width, total_height), (255, 255, 255))
        labeled.paste(image, (0, label_height))
    else:
        labeled = Image.new('RGB', (width, total_height), (255, 255, 255))
        labeled.paste(image, (0, 0))
    
    # Draw label
    draw = ImageDraw.Draw(labeled)
    
    # Try to use a default font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position (centered)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    if position == 'top':
        x = (width - text_width) // 2
        y = (label_height - text_height) // 2
    else:
        x = (width - text_width) // 2
        y = height + (label_height - text_height) // 2
    
    draw.text((x, y), label, fill=(0, 0, 0), font=font)
    
    return labeled
