"""Tests for configuration management."""

import pytest
import yaml
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from style_shift.core.config import (
    Config,
    load_config,
    save_config,
    get_default_config,
)


class TestConfigDataclass:
    """Tests for Config dataclass."""
    
    def test_default_values(self):
        """Test that default configuration has expected values."""
        config = Config()
        assert config.alpha == 1.0
        assert config.size == 512
        assert config.device is None
        assert config.style_name is None
        assert config.preserve_color is False
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = Config(alpha=0.8, size=768, device='cuda')
        assert config.alpha == 0.8
        assert config.size == 768
        assert config.device == 'cuda'
    
    def test_invalid_alpha_raises_error(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be between"):
            Config(alpha=1.5)
        
        with pytest.raises(ValueError, match="alpha must be between"):
            Config(alpha=-0.1)
    
    def test_invalid_size_raises_error(self):
        """Test that invalid size raises ValueError."""
        with pytest.raises(ValueError, match="size must be positive"):
            Config(size=0)
        
        with pytest.raises(ValueError, match="size must be positive"):
            Config(size=-100)
    
    def test_invalid_device_raises_error(self):
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="device must be"):
            Config(device='invalid')
    
    def test_invalid_crop_raises_error(self):
        """Test that invalid crop raises ValueError."""
        with pytest.raises(ValueError, match="crop must be"):
            Config(crop='invalid')
    
    def test_get_device_returns_torch_device(self):
        """Test that get_device() returns torch.device."""
        config = Config(device='cpu')
        device = config.get_device()
        assert isinstance(device, torch.device)
        assert device.type == 'cpu'
    
    def test_to_dict_returns_dict(self):
        """Test that to_dict() returns dictionary."""
        config = Config(alpha=0.5, size=256)
        data = config.to_dict()
        assert isinstance(data, dict)
        assert data['alpha'] == 0.5
        assert data['size'] == 256
    
    def test_from_dict_creates_config(self):
        """Test that from_dict() creates Config from dictionary."""
        data = {'alpha': 0.7, 'size': 1024, 'device': 'cpu'}
        config = Config.from_dict(data)
        assert config.alpha == 0.7
        assert config.size == 1024
        assert config.device == 'cpu'


class TestGetDefaultConfig:
    """Tests for get_default_config()."""
    
    def test_returns_config_instance(self):
        """Test that function returns Config instance."""
        config = get_default_config()
        assert isinstance(config, Config)
    
    def test_has_default_values(self):
        """Test that returned config has default values."""
        config = get_default_config()
        assert config.alpha == 1.0
        assert config.size == 512


class TestLoadConfig:
    """Tests for load_config()."""
    
    def test_load_default_when_none(self):
        """Test that load_config(None) returns default config."""
        config = load_config(None)
        assert isinstance(config, Config)
        assert config.alpha == 1.0
    
    def test_load_from_yaml_file(self, tmp_path):
        """Test loading config from YAML file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("alpha: 0.6\nsize: 768\ndevice: cpu")
        
        config = load_config(str(yaml_file))
        
        assert config.alpha == 0.6
        assert config.size == 768
        assert config.device == 'cpu'
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")
    
    def test_load_empty_yaml_returns_default(self, tmp_path):
        """Test that empty YAML file returns default config."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        
        config = load_config(str(yaml_file))
        
        assert config.alpha == 1.0
        assert config.size == 512


class TestSaveConfig:
    """Tests for save_config()."""
    
    def test_save_creates_file(self, tmp_path):
        """Test that save_config creates YAML file."""
        config = Config(alpha=0.8, size=256)
        output_file = tmp_path / "output.yaml"
        
        save_config(config, str(output_file))
        
        assert output_file.exists()
    
    def test_save_creates_parent_directories(self, tmp_path):
        """Test that save_config creates parent directories."""
        config = Config()
        output_file = tmp_path / "nested" / "dir" / "config.yaml"
        
        save_config(config, str(output_file))
        
        assert output_file.exists()
    
    def test_save_then_load_roundtrip(self, tmp_path):
        """Test that saved config can be loaded back."""
        original = Config(alpha=0.75, size=1024, device='cpu', preserve_color=True)
        output_file = tmp_path / "roundtrip.yaml"
        
        save_config(original, str(output_file))
        loaded = load_config(str(output_file))
        
        assert loaded.alpha == original.alpha
        assert loaded.size == original.size
        assert loaded.device == original.device
        assert loaded.preserve_color == original.preserve_color
    
    def test_saved_yaml_is_valid(self, tmp_path):
        """Test that saved YAML file can be parsed."""
        config = Config(alpha=0.9, size=512)
        output_file = tmp_path / "valid.yaml"
        
        save_config(config, str(output_file))
        
        with open(output_file, 'r') as f:
            data = yaml.safe_load(f)
        
        assert isinstance(data, dict)
        assert data['alpha'] == 0.9
        assert data['size'] == 512
