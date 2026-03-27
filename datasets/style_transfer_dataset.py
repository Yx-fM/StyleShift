"""Style Transfer Dataset for training StyleShift Decoder."""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms
import random


class StyleTransferDataset(Dataset):
    """PyTorch Dataset for style transfer training.
    
    Randomly pairs content images (MS-COCO) with style images (WikiArt).
    Applies data augmentation for better generalization.
    
    Args:
        content_root: Path to MS-COCO content images directory
        style_root: Path to WikiArt style images directory
        image_size: Output image size (square)
        content_transform: Transform for content images
        style_transform: Transform for style images
    
    Returns:
        Tuple of (content_image, style_image) tensors
    """
    
    def __init__(
        self,
        content_root: str,
        style_root: str,
        image_size: int = 256,
        content_transform=None,
        style_transform=None
    ):
        self.content_root = Path(content_root)
        self.style_root = Path(style_root)
        self.image_size = image_size
        
        # Default transforms
        if content_transform is None:
            self.content_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.content_transform = content_transform
        
        if style_transform is None:
            self.style_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.style_transform = style_transform
        
        # Collect image paths
        print("Loading content images...")
        self.content_images = self._collect_images(content_root)
        print(f"  Found {len(self.content_images)} content images")
        
        print("Loading style images...")
        self.style_images = self._collect_images(style_root)
        print(f"  Found {len(self.style_images)} style images")
    
    def _collect_images(self, root: Path):
        """Collect all image paths from directory."""
        root = Path(root)  # Ensure it's a Path object
        if not root.exists():
            print(f"  Warning: Directory not found: {root}")
            return []
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(root.rglob(ext))
        
        return images
    
    def __len__(self):
        """Return the maximum of content/style images."""
        return max(len(self.content_images), len(self.style_images))
    
    def __getitem__(self, idx):
        """Get a random content-style pair."""
        # Random pair
        content_idx = random.randint(0, len(self.content_images) - 1)
        style_idx = random.randint(0, len(self.style_images) - 1)
        
        # Load images
        try:
            content_img = Image.open(self.content_images[content_idx]).convert('RGB')
        except Exception as e:
            print(f"Error loading content image: {e}")
            content_img = Image.new('RGB', (self.image_size, self.image_size))
        
        try:
            style_img = Image.open(self.style_images[style_idx]).convert('RGB')
        except Exception as e:
            print(f"Error loading style image: {e}")
            style_img = Image.new('RGB', (self.image_size, self.image_size))
        
        # Apply transforms
        content_tensor = self.content_transform(content_img)
        style_tensor = self.style_transform(style_img)
        
        return content_tensor, style_tensor
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for data loading."""
        content_batch, style_batch = zip(*batch)
        return torch.stack(content_batch), torch.stack(style_batch)
