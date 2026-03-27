"""Training configuration for StyleShift Decoder."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


@dataclass
class TrainingConfig:
    """Training configuration dataclass.
    
    Attributes:
        # Dataset settings
        content_root: Path to MS-COCO content images
        style_root: Path to WikiArt style images
        image_size: Training image size (square)
        
        # Training hyperparameters
        epochs: Number of training epochs
        batch_size: Batch size per iteration
        learning_rate: Initial learning rate
        lr_decay: Learning rate decay factor per iteration
        
        # Loss weights
        content_weight: Content loss weight (alpha)
        style_weight: Style loss weight (beta)
        tv_weight: Total variation loss weight (gamma)
        
        # Optimizer settings
        betas: Adam optimizer betas (beta1, beta2)
        
        # Checkpointing
        save_dir: Directory to save checkpoints
        save_every: Save checkpoint every N epochs
        keep_last: Keep last N checkpoints
        
        # Monitoring
        log_dir: TensorBoard log directory
        log_every: Log metrics every N steps
        validate_every: Validate every N epochs
        
        # Data loading
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    """
    
    # Dataset settings
    content_root: str = "data/coco"
    style_root: str = "data/wikiart"
    image_size: int = 256
    
    # Training hyperparameters
    epochs: int = 16
    batch_size: int = 8
    learning_rate: float = 1e-4
    lr_decay: float = 5e-5  # Per iteration decay
    
    # Loss weights (AdaIN official: style:content = 10:1)
    content_weight: float = 1.0
    style_weight: float = 10.0
    tv_weight: float = 1e-6
    
    # Optimizer settings
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 1  # epochs
    keep_last: int = 3
    
    # Monitoring
    log_dir: str = "runs"
    log_every: int = 10  # steps
    validate_every: int = 1  # epochs
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Validation
    val_size: int = 10  # Number of validation samples
    
    def __post_init__(self):
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
        
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        
        if not (0.0 <= self.content_weight <= 100.0):
            raise ValueError(f"Content weight out of range: {self.content_weight}")
        
        if not (0.0 <= self.style_weight <= 100.0):
            raise ValueError(f"Style weight out of range: {self.style_weight}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
