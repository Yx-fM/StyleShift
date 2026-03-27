"""Image input/output utilities for loading, saving, and tensor conversion."""

import torch
from PIL import Image
from torchvision import transforms
from typing import Union, Optional
from pathlib import Path


# Default image transformations
def get_default_transform():
    """Get default image transformation pipeline.
    
    Returns:
        transforms.Compose: Transformation pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range and C,H,W format
    ])


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to torch.Tensor.
    
    Args:
        image: PIL Image to convert.
    
    Returns:
        torch.Tensor: Image tensor in shape (C, H, W) with values in [0, 1].
    
    Examples:
        >>> from PIL import Image
        >>> img = Image.open('test.jpg')
        >>> tensor = pil_to_tensor(img)
        >>> print(tensor.shape)
        torch.Size([3, 512, 512])
    """
    transform = get_default_transform()
    return transform(image)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert torch.Tensor to PIL Image.
    
    Args:
        tensor: Image tensor in shape (C, H, W) or (1, C, H, W) with values in [0, 1].
    
    Returns:
        Image.Image: PIL Image in RGB mode.
    
    Examples:
        >>> tensor = torch.rand(3, 224, 224)
        >>> img = tensor_to_pil(tensor)
        >>> print(img.mode)
        RGB
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


def load_image(
    path: Union[str, Path],
    mode: str = 'RGB',
    max_size: Optional[int] = None
) -> torch.Tensor:
    """Load image from file and convert to tensor.
    
    Args:
        path: Path to image file.
        mode: PIL mode (e.g., 'RGB', 'RGBA', 'L'). Default: 'RGB'.
        max_size: Maximum dimension. If specified, image will be resized to fit.
    
    Returns:
        torch.Tensor: Image tensor in shape (C, H, W) with values in [0, 1].
    
    Raises:
        FileNotFoundError: If image file doesn't exist.
        ValueError: If image cannot be loaded.
    
    Examples:
        >>> tensor = load_image('photo.jpg')
        >>> print(tensor.shape)
        torch.Size([3, 512, 512])
        
        >>> tensor = load_image('photo.png', max_size=256)
        >>> print(tensor.shape)
        torch.Size([3, 256, 256])
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    try:
        image = Image.open(path)
        
        # Convert to specified mode
        if mode:
            image = image.convert(mode)
        
        # Resize if max_size specified
        if max_size is not None:
            width, height = image.size
            if max(width, height) > max_size:
                scale = max_size / max(width, height)
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return pil_to_tensor(image)
    
    except Exception as e:
        raise ValueError(f"Failed to load image: {path}. Error: {e}")


def save_image(
    tensor: torch.Tensor,
    path: Union[str, Path],
    quality: int = 95
) -> None:
    """Save image tensor to file.
    
    Args:
        tensor: Image tensor in shape (C, H, W) or (1, C, H, W) with values in [0, 1].
        path: Output file path.
        quality: JPEG quality (1-100). Default: 95.
    
    Raises:
        ValueError: If tensor shape is invalid.
    
    Examples:
        >>> tensor = torch.rand(3, 512, 512)
        >>> save_image(tensor, 'output.jpg')
        
        >>> tensor = torch.rand(1, 3, 224, 224)
        >>> save_image(tensor, 'output.png')
    """
    path = Path(path)
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensor to PIL
    image = tensor_to_pil(tensor)
    
    # Save based on extension
    save_kwargs = {}
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        save_kwargs['quality'] = quality
        save_kwargs['optimize'] = True
    
    image.save(path, **save_kwargs)


def normalize(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """Normalize tensor with given mean and standard deviation.
    
    Args:
        tensor: Image tensor in shape (C, H, W).
        mean: List of mean values for each channel.
        std: List of standard deviation values for each channel.
    
    Returns:
        torch.Tensor: Normalized tensor.
    
    Examples:
        >>> tensor = torch.rand(3, 224, 224)
        >>> normalized = normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor (C, H, W), got {tensor.dim()}D")
    
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    
    return (tensor - mean) / std


def denormalize(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """Denormalize tensor with given mean and standard deviation.
    
    Args:
        tensor: Normalized image tensor in shape (C, H, W).
        mean: List of mean values used for normalization.
        std: List of standard deviation values used for normalization.
    
    Returns:
        torch.Tensor: Denormalized tensor in [0, 1] range.
    
    Examples:
        >>> tensor = torch.randn(3, 224, 224)
        >>> denormalized = denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor (C, H, W), got {tensor.dim()}D")
    
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    
    result = tensor * std + mean
    return result.clamp(0, 1)
