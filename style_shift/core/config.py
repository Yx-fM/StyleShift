"""Configuration management with dataclasses and YAML support."""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Union
from pathlib import Path

import yaml
import torch

from style_shift.utils.device import get_device


@dataclass
class Config:
    """Configuration dataclass for StyleShift.
    
    Attributes:
        alpha: Style strength (0.0-1.0). Default: 1.0
        size: Maximum output dimension. Default: 512
        device: Device specification ('cuda', 'cpu', 'mps', or None for auto). Default: None
        style_name: Name of built-in style to use. Default: None
        content_path: Path to content image. Default: None
        style_path: Path to style image. Default: None
        output_path: Path to save output image. Default: None
        preserve_color: Whether to preserve content colors. Default: False
        crop: Crop strategy ('center', 'random', or None). Default: None
    """
    
    # Style transfer parameters
    alpha: float = 1.0
    size: int = 512
    device: Optional[str] = None
    
    # Image paths
    style_name: Optional[str] = None
    content_path: Optional[str] = None
    style_path: Optional[str] = None
    output_path: Optional[str] = None
    
    # Processing options
    preserve_color: bool = False
    crop: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {self.alpha}")
        
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        
        if self.device is not None and self.device not in ['cuda', 'cpu', 'mps']:
            raise ValueError(f"device must be 'cuda', 'cpu', 'mps', or None, got {self.device}")
        
        if self.crop is not None and self.crop not in ['center', 'random']:
            raise ValueError(f"crop must be 'center', 'random', or None, got {self.crop}")
    
    def get_device(self) -> torch.device:
        """Get the torch.device for this configuration.
        
        Returns:
            torch.device: The device to use for computation.
        """
        return get_device(self.device)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            dict: Configuration as dictionary.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Config':
        """Create config from dictionary.
        
        Args:
            data: Dictionary with configuration values.
        
        Returns:
            Config: Configuration object.
        """
        return cls(**data)


def get_default_config() -> Config:
    """Get default configuration.
    
    Returns:
        Config: Default configuration object.
    """
    return Config()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from YAML file or return defaults.
    
    Args:
        config_path: Path to YAML configuration file. If None, returns default config.
    
    Returns:
        Config: Configuration object.
    
    Examples:
        >>> config = load_config()  # Default config
        >>> config = load_config('config.yaml')  # Load from file
    """
    if config_path is None:
        return get_default_config()
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        return get_default_config()
    
    return Config.from_dict(data)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        config_path: Path to save YAML file.
    
    Examples:
        >>> config = Config(alpha=0.8, size=768)
        >>> save_config(config, 'config.yaml')
    """
    config_path = Path(config_path)
    
    # Create parent directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
