"""Pytest fixtures for StyleShift test suite."""

import pytest
import torch
from PIL import Image
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.new('RGB', (224, 224), color='red')


@pytest.fixture
def sample_tensor():
    """Create a sample image tensor for testing."""
    return torch.rand(3, 224, 224)


@pytest.fixture
def temp_image_file(tmp_path, sample_image):
    """Create a temporary image file."""
    img_file = tmp_path / "test.jpg"
    sample_image.save(img_file)
    return img_file


@pytest.fixture
def sample_jpg(tmp_path):
    """Create a sample JPEG file."""
    img_file = tmp_path / "sample.jpg"
    img = Image.new('RGB', (512, 512), color='blue')
    img.save(img_file)
    return img_file


@pytest.fixture
def sample_png(tmp_path):
    """Create a sample PNG file with alpha channel."""
    img_file = tmp_path / "sample.png"
    img = Image.new('RGBA', (512, 512), color=(0, 255, 0, 128))
    img.save(img_file)
    return img_file


@pytest.fixture
def trained_model():
    """Create a mock trained model for testing."""
    return torch.nn.Conv2d(3, 64, kernel_size=3)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for model manager tests."""
    return str(tmp_path / "cache")
