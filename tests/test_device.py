"""Tests for device management utilities."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from style_shift.utils.device import (
    get_device,
    is_cuda_available,
    to_device,
    get_device_name,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)


class TestIsCudaAvailable:
    """Tests for is_cuda_available()."""
    
    def test_returns_boolean(self):
        """Test that function returns a boolean value."""
        result = is_cuda_available()
        assert isinstance(result, bool)
    
    def test_matches_torch_cuda_is_available(self):
        """Test that function matches torch.cuda.is_available()."""
        result = is_cuda_available()
        expected = torch.cuda.is_available()
        assert result == expected


class TestGetDevice:
    """Tests for get_device()."""
    
    def test_returns_torch_device(self):
        """Test that get_device returns a torch.device object."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_auto_selects_cuda_when_available(self):
        """Test that auto-selection prefers CUDA when available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = get_device()
                assert device.type == 'cuda'
    
    def test_auto_selects_cpu_when_no_cuda(self):
        """Test that auto-selection falls back to CPU when CUDA unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = get_device()
                assert device.type == 'cpu'
    
    def test_respects_cuda_preference(self):
        """Test that preferred='cuda' is respected."""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device(preferred='cuda')
            assert device.type == 'cuda'
    
    def test_falls_back_to_cpu_when_cuda_preferred_but_unavailable(self):
        """Test that preferred='cuda' falls back to CPU when CUDA unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            device = get_device(preferred='cuda')
            assert device.type == 'cpu'
    
    def test_respects_cpu_preference(self):
        """Test that preferred='cpu' is respected."""
        device = get_device(preferred='cpu')
        assert device.type == 'cpu'
    
    def test_invalid_device_raises_error(self):
        """Test that invalid device preference raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device type"):
            get_device(preferred='invalid')


class TestToDevice:
    """Tests for to_device()."""
    
    def test_moves_model_to_device(self):
        """Test that to_device moves model to specified device."""
        model = SimpleModel()
        device = torch.device('cpu')
        
        moved_model = to_device(model, device)
        
        # Check model parameters are on correct device
        for param in moved_model.parameters():
            assert param.device.type == 'cpu'
    
    def test_auto_detects_device_when_none_specified(self):
        """Test that to_device auto-detects when device=None."""
        model = SimpleModel()
        
        with patch('style_shift.utils.device.get_device') as mock_get:
            mock_get.return_value = torch.device('cpu')
            moved_model = to_device(model)
            
            mock_get.assert_called_once()
            for param in moved_model.parameters():
                assert param.device.type == 'cpu'
    
    def test_accepts_string_device(self):
        """Test that to_device accepts string device specification."""
        model = SimpleModel()
        moved_model = to_device(model, 'cpu')
        
        for param in moved_model.parameters():
            assert param.device.type == 'cpu'
    
    def test_returns_same_model_type(self):
        """Test that to_device returns the same model type."""
        model = SimpleModel()
        moved_model = to_device(model, 'cpu')
        
        assert isinstance(moved_model, SimpleModel)


class TestGetDeviceName:
    """Tests for get_device_name()."""
    
    def test_returns_string_for_cpu(self):
        """Test that get_device_name returns string for CPU."""
        device = torch.device('cpu')
        name = get_device_name(device)
        assert isinstance(name, str)
        assert 'CPU' in name
    
    def test_returns_string_for_cuda(self):
        """Test that get_device_name returns string for CUDA."""
        device = torch.device('cuda:0')
        
        with patch('torch.cuda.get_device_name', return_value='Test GPU'):
            name = get_device_name(device)
            assert isinstance(name, str)
            assert 'Test GPU' in name
    
    def test_auto_detects_current_device(self):
        """Test that get_device_name auto-detects when device=None."""
        with patch('style_shift.utils.device.get_device') as mock_get:
            mock_get.return_value = torch.device('cpu')
            name = get_device_name()
            
            mock_get.assert_called_once()
            assert isinstance(name, str)
