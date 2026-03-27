"""Model management utilities for downloading, caching, and version control."""

import hashlib
import json
import os
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .device import get_device


class ModelManager:
    """Manager for downloading and caching pre-trained models.
    
    Attributes:
        cache_dir: Directory to store downloaded models.
        models_info: Dictionary of available models and their metadata.
    """
    
    DEFAULT_MODELS: Dict[str, Dict[str, Any]] = {
        'decoder_anime': {
            'url': 'https://example.com/models/decoder_anime.pth',
            'hash': 'abc123',  # SHA256 hash for verification
            'size_mb': 50,
            'description': 'Anime style decoder'
        },
        'decoder_vangogh': {
            'url': 'https://example.com/models/decoder_vangogh.pth',
            'hash': 'def456',
            'size_mb': 50,
            'description': 'Van Gogh style decoder'
        },
        'decoder_monet': {
            'url': 'https://example.com/models/decoder_monet.pth',
            'hash': 'ghi789',
            'size_mb': 50,
            'description': 'Monet style decoder'
        },
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize ModelManager.
        
        Args:
            cache_dir: Directory to cache models. If None, uses:
                      - ~/.styleshift/models on Linux/macOS
                      - %USERPROFILE%\\.styleshift\\models on Windows
        """
        if cache_dir is None:
            # Default cache directory
            home = Path.home()
            cache_dir = home / ".styleshift" / "models"
        
        self.cache_dir = Path(cache_dir)
        self.models_info = self.DEFAULT_MODELS.copy()
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the path where a model should be cached.
        
        Args:
            model_name: Name of the model.
        
        Returns:
            Path: Path to the cached model file.
        """
        return self.cache_dir / f"{model_name}.pth"
    
    def is_model_cached(self, model_name: str) -> bool:
        """Check if a model is already cached.
        
        Args:
            model_name: Name of the model.
        
        Returns:
            bool: True if model is cached, False otherwise.
        """
        model_path = self.get_model_path(model_name)
        return model_path.exists()
    
    def download_model(
        self,
        model_name: str,
        force: bool = False,
        verify_hash: bool = True
    ) -> Path:
        """Download a model to the cache directory.
        
        Args:
            model_name: Name of the model to download.
            force: If True, re-download even if already cached.
            verify_hash: If True, verify SHA256 hash after download.
        
        Returns:
            Path: Path to the downloaded model file.
        
        Raises:
            KeyError: If model_name is not in available models.
            RuntimeError: If download fails or hash verification fails.
        
        Examples:
            >>> mm = ModelManager()
            >>> path = mm.download_model('decoder_anime')
            >>> print(path)
            ~/.styleshift/models/decoder_anime.pth
        """
        if model_name not in self.models_info:
            raise KeyError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self.models_info.keys())}"
            )
        
        model_path = self.get_model_path(model_name)
        
        # Check if already cached
        if model_path.exists() and not force:
            print(f"Model '{model_name}' already cached at {model_path}")
            return model_path
        
        # Download model
        model_info = self.models_info[model_name]
        url = model_info['url']
        expected_hash = model_info.get('hash')
        
        print(f"Downloading {model_name} from {url}...")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save to file
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify hash
            if verify_hash and expected_hash:
                actual_hash = self._compute_hash(model_path)
                if actual_hash != expected_hash:
                    model_path.unlink()  # Remove corrupted file
                    raise RuntimeError(
                        f"Hash verification failed for {model_name}. "
                        f"Expected: {expected_hash}, Got: {actual_hash}"
                    )
            
            print(f"Successfully downloaded {model_name} to {model_path}")
            return model_path
        
        except requests.RequestException as e:
            if model_path.exists():
                model_path.unlink()  # Clean up partial download
            raise RuntimeError(f"Failed to download {model_name}: {e}")
    
    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file.
        
        Args:
            file_path: Path to the file.
        
        Returns:
            str: SHA256 hash as hexadecimal string.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def list_available_models(self) -> List[str]:
        """List all available models for download.
        
        Returns:
            List[str]: List of model names.
        
        Examples:
            >>> mm = ModelManager()
            >>> models = mm.list_available_models()
            >>> print(models)
            ['decoder_anime', 'decoder_vangogh', 'decoder_monet']
        """
        return list(self.models_info.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model.
        
        Returns:
            Dict containing model metadata (url, hash, size_mb, description).
        
        Raises:
            KeyError: If model_name is not found.
        """
        if model_name not in self.models_info:
            raise KeyError(f"Model '{model_name}' not found")
        
        return self.models_info[model_name].copy()
    
    def clear_cache(self) -> None:
        """Clear all cached models.
        
        Warning: This will delete all downloaded models.
        """
        if self.cache_dir.exists():
            for file in self.cache_dir.glob("*.pth"):
                file.unlink()
            print(f"Cleared cache directory: {self.cache_dir}")
    
    def get_cache_size(self) -> int:
        """Get total size of cached models in bytes.
        
        Returns:
            int: Total size in bytes.
        """
        total_size = 0
        for file in self.cache_dir.glob("*.pth"):
            total_size += file.stat().st_size
        return total_size
