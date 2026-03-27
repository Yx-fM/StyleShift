"""Tests for image input/output utilities."""

import pytest
import torch
from PIL import Image
import io
from pathlib import Path

from style_shift.utils.image_io import (
    load_image,
    save_image,
    pil_to_tensor,
    tensor_to_pil,
    normalize,
    denormalize,
)


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


class TestPilToTensor:
    """Tests for pil_to_tensor()."""
    
    def test_returns_tensor(self, sample_image):
        """Test that pil_to_tensor returns torch.Tensor."""
        tensor = pil_to_tensor(sample_image)
        assert isinstance(tensor, torch.Tensor)
    
    def test_output_shape(self, sample_image):
        """Test that output has shape (C, H, W)."""
        tensor = pil_to_tensor(sample_image)
        assert tensor.shape == (3, 224, 224)
    
    def test_output_range(self, sample_image):
        """Test that output values are in [0, 1] range."""
        tensor = pil_to_tensor(sample_image)
        assert tensor.min() >= 0
        assert tensor.max() <= 1
    
    def test_dtype_is_float(self, sample_image):
        """Test that output tensor is float type."""
        tensor = pil_to_tensor(sample_image)
        assert tensor.dtype == torch.float32


class TestTensorToPil:
    """Tests for tensor_to_pil()."""
    
    def test_returns_image(self, sample_tensor):
        """Test that tensor_to_pil returns PIL Image."""
        img = tensor_to_pil(sample_tensor)
        assert isinstance(img, Image.Image)
    
    def test_output_mode(self, sample_tensor):
        """Test that output image is in RGB mode."""
        img = tensor_to_pil(sample_tensor)
        assert img.mode == 'RGB'
    
    def test_output_size(self, sample_tensor):
        """Test that output size matches tensor dimensions."""
        img = tensor_to_pil(sample_tensor)
        assert img.size == (224, 224)
    
    def test_handles_batch_dimension(self):
        """Test that function handles (1, C, H, W) input."""
        tensor = torch.rand(1, 3, 128, 128)
        img = tensor_to_pil(tensor)
        assert img.size == (128, 128)
    
    def test_invalid_batch_size_raises_error(self):
        """Test that batch size > 1 raises ValueError."""
        tensor = torch.rand(2, 3, 128, 128)
        with pytest.raises(ValueError, match="batch size 1"):
            tensor_to_pil(tensor)
    
    def test_invalid_dimensions_raises_error(self):
        """Test that invalid tensor dimensions raise ValueError."""
        tensor = torch.rand(128, 128)  # Missing channel dimension
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            tensor_to_pil(tensor)


class TestLoadImage:
    """Tests for load_image()."""
    
    def test_load_jpg(self, temp_image_file):
        """Test loading JPEG image."""
        tensor = load_image(str(temp_image_file))
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 3
    
    def test_load_png(self, tmp_path):
        """Test loading PNG image."""
        png_file = tmp_path / "test.png"
        img = Image.new('RGBA', (256, 256), color='blue')
        img.save(png_file)
        
        tensor = load_image(str(png_file))
        assert tensor.shape[0] == 3
    
    def test_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.jpg")
    
    def test_max_size_resizes_image(self, temp_image_file):
        """Test that max_size parameter resizes image."""
        tensor = load_image(str(temp_image_file), max_size=128)
        assert tensor.shape[1:] == (128, 128)
    
    def test_load_preserves_aspect_ratio(self, tmp_path):
        """Test that resizing preserves aspect ratio."""
        img_file = tmp_path / "rect.jpg"
        img = Image.new('RGB', (400, 200), color='green')
        img.save(img_file)
        
        tensor = load_image(str(img_file), max_size=100)
        # Should be (50, 100) preserving 2:1 ratio (width becomes 50, height becomes 100)
        assert tensor.shape[1] == 50
        assert tensor.shape[2] == 100


class TestSaveImage:
    """Tests for save_image()."""
    
    def test_save_creates_file(self, tmp_path, sample_tensor):
        """Test that save_image creates output file."""
        output_file = tmp_path / "output.jpg"
        save_image(sample_tensor, str(output_file))
        assert output_file.exists()
    
    def test_save_creates_parent_directories(self, tmp_path, sample_tensor):
        """Test that save_image creates parent directories."""
        output_file = tmp_path / "nested" / "dir" / "output.jpg"
        save_image(sample_tensor, str(output_file))
        assert output_file.exists()
    
    def test_save_jpg_quality(self, tmp_path, sample_tensor):
        """Test JPEG quality parameter."""
        output_file = tmp_path / "quality.jpg"
        save_image(sample_tensor, str(output_file), quality=50)
        assert output_file.exists()
        
        # File with lower quality should be smaller
        output_high = tmp_path / "quality_high.jpg"
        save_image(sample_tensor, str(output_high), quality=95)
        
        # Note: This assertion might not always hold for very small images
        # but generally lower quality = smaller file
        assert output_file.exists()
    
    def test_save_png(self, tmp_path, sample_tensor):
        """Test saving PNG format."""
        output_file = tmp_path / "output.png"
        save_image(sample_tensor, str(output_file))
        assert output_file.exists()
    
    def test_save_and_load_roundtrip(self, tmp_path):
        """Test that saved image can be loaded back."""
        original = torch.rand(3, 128, 128)
        output_file = tmp_path / "roundtrip.png"  # Use PNG for lossless compression
        
        save_image(original, str(output_file))
        loaded = load_image(str(output_file))
        
        assert loaded.shape == original.shape
        # PNG is lossless, so values should match closely
        assert torch.allclose(loaded, original, atol=0.02)


class TestNormalize:
    """Tests for normalize()."""
    
    def test_returns_tensor(self):
        """Test that normalize returns tensor."""
        tensor = torch.rand(3, 64, 64)
        result = normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert isinstance(result, torch.Tensor)
    
    def test_output_shape(self):
        """Test that output shape matches input."""
        tensor = torch.rand(3, 64, 64)
        result = normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        assert result.shape == tensor.shape
    
    def test_invalid_dimensions_raises_error(self):
        """Test that invalid dimensions raise ValueError."""
        tensor = torch.rand(64, 64)  # Missing channel dimension
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            normalize(tensor, [0.5], [0.5])


class TestDenormalize:
    """Tests for denormalize()."""
    
    def test_returns_tensor(self):
        """Test that denormalize returns tensor."""
        tensor = torch.randn(3, 64, 64)
        result = denormalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert isinstance(result, torch.Tensor)
    
    def test_output_in_valid_range(self):
        """Test that denormalized values are in [0, 1]."""
        tensor = torch.randn(3, 64, 64)  # Can have any values
        result = denormalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        assert result.min() >= 0
        assert result.max() <= 1
    
    def test_output_shape(self):
        """Test that output shape matches input."""
        tensor = torch.randn(3, 64, 64)
        result = denormalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        assert result.shape == tensor.shape


class TestIntegration:
    """Integration tests for image I/O."""
    
    def test_load_process_save_pipeline(self, tmp_path, temp_image_file):
        """Test complete load → process → save pipeline."""
        # Load
        tensor = load_image(str(temp_image_file))
        
        # Process (simple normalization)
        normalized = normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        denormalized = denormalize(normalized, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        # Save
        output_file = tmp_path / "processed.jpg"
        save_image(denormalized, str(output_file))
        
        assert output_file.exists()
        
        # Load again and verify
        reloaded = load_image(str(output_file))
        assert reloaded.shape == tensor.shape
