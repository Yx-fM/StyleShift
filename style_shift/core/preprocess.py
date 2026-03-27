"""Image preprocessing utilities for style transfer."""

import torch
from PIL import Image
from pathlib import Path
from typing import Union, Optional
from torchvision import transforms

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_default_transform():
    """Get default image transformation pipeline."""
    return transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range and C,H,W format
    ])


def get_normalized_transform():
    """Get transformation with ImageNet normalization."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def resize_image(
    image: Image.Image,
    max_size: int,
    maintain_aspect: bool = True
) -> Image.Image:
    """Resize image while maintaining aspect ratio.
    
    Args:
        image: PIL Image to resize
        max_size: Maximum dimension (width or height)
        maintain_aspect: Whether to maintain aspect ratio. Default: True
    
    Returns:
        PIL.Image: Resized image
    """
    if not maintain_aspect:
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
    
    width, height = image.size
    
    # No resize needed if already smaller than max_size
    if max(width, height) <= max_size:
        return image
    
    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    """Apply center crop to image.
    
    Args:
        image: PIL Image to crop
        crop_size: Size of the square crop
    
    Returns:
        PIL.Image: Center-cropped image
    """
    width, height = image.size
    
    # Calculate crop coordinates
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    return image.crop((left, top, right, bottom))


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to torch.Tensor.
    
    Args:
        image: PIL Image to convert
    
    Returns:
        torch.Tensor: Image tensor in shape (C, H, W) with values in [0, 1]
    """
    transform = get_default_transform()
    return transform(image)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert torch.Tensor to PIL Image.
    
    Args:
        tensor: Image tensor in shape (C, H, W) or (1, C, H, W) with values in [0, 1]
    
    Returns:
        Image.Image: PIL Image in RGB mode
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        else:
            raise ValueError(f"Expected batch size 1, got {tensor.shape[0]}")
    
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor (C, H, W), got {tensor.dim()}D")
    
    # Ensure tensor is in [0, 1] range
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL Image
    transform = transforms.ToPILImage()
    return transform(tensor)


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor with ImageNet statistics.
    
    Args:
        tensor: Image tensor in shape (C, H, W) or (1, C, H, W)
    
    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return (tensor - mean) / std


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor from ImageNet statistics to [0, 1] range.
    
    Args:
        tensor: Normalized image tensor in shape (C, H, W) or (1, C, H, W)
    
    Returns:
        torch.Tensor: Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    result = tensor * std + mean
    return result.clamp(0, 1)


def preprocess_image(
    image: Union[Image.Image, torch.Tensor, str, Path],
    max_size: int = 512,
    normalize: bool = True,
    mode: str = 'RGB'
) -> torch.Tensor:
    """Preprocess image for style transfer.
    
    Args:
        image: Input image (PIL, Tensor, file path, or URL)
        max_size: Maximum dimension. If specified, image will be resized. Default: 512
        normalize: Whether to apply ImageNet normalization. Default: True
        mode: PIL mode (e.g., 'RGB', 'RGBA', 'L'). Default: 'RGB'
    
    Returns:
        torch.Tensor: Preprocessed image tensor in shape (C, H, W)
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    # Handle different input types
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        image = Image.open(path)
    
    elif isinstance(image, torch.Tensor):
        if image.dim() == 4:
            if image.shape[0] == 1:
                image = tensor_to_pil(image[0])
            else:
                raise ValueError(f"Expected batch size 1, got {image.shape[0]}")
        else:
            image = tensor_to_pil(image)
    
    # Convert to specified mode (usually RGB)
    if mode and image.mode != mode:
        if mode == 'RGB' and image.mode == 'RGBA':
            # Handle RGBA by compositing on white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background
        else:
            image = image.convert(mode)
    
    # Resize if max_size specified
    if max_size is not None:
        image = resize_image(image, max_size, maintain_aspect=True)
    
    # Convert to tensor
    if normalize:
        transform = get_normalized_transform()
        tensor = transform(image)
    else:
        tensor = pil_to_tensor(image)
    
    return tensor


def prepare_content_tensor(
    content: Union[str, Path, Image.Image, torch.Tensor],
    max_size: int = 512
) -> torch.Tensor:
    """Prepare content image tensor (normalized, shaped for VGG).
    
    Args:
        content: Content image (path, PIL, or tensor)
        max_size: Maximum dimension. Default: 512
    
    Returns:
        torch.Tensor: Content tensor in shape (1, C, H, W) ready for VGG
    """
    tensor = preprocess_image(content, max_size=max_size, normalize=True)
    # Add batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def prepare_style_tensor(
    style: Union[str, Path, Image.Image, torch.Tensor],
    max_size: int = 512
) -> torch.Tensor:
    """Prepare style image tensor (normalized, shaped for VGG).
    
    Args:
        style: Style image (path, PIL, or tensor)
        max_size: Maximum dimension. Default: 512
    
    Returns:
        torch.Tensor: Style tensor in shape (1, C, H, W) ready for VGG
    """
    tensor = preprocess_image(style, max_size=max_size, normalize=True)
    # Add batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


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
        torch.Tensor: Blended image
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
    preserve_ratio = max(0.0, min(1.0, preserve_ratio))
    return blend_images(stylized, original, preserve_ratio)
