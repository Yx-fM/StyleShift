"""Tests for AdaIN (Adaptive Instance Normalization) layer."""

import pytest
import torch
import torch.nn as nn

from style_shift.models.adain import AdaIN, adain_function


class TestAdaINFunction:
    """Tests for adain_function()."""
    
    def test_output_shape_matches_input(self):
        """Test that output shape matches content input shape."""
        content = torch.randn(2, 512, 14, 14)
        style = torch.randn(2, 512, 14, 14)
        output = adain_function(content, style)
        assert output.shape == content.shape
    
    def test_output_mean_matches_style_mean(self):
        """Test that output channel means match style means."""
        content = torch.randn(1, 512, 14, 14)
        style = torch.randn(1, 512, 14, 14)
        
        output = adain_function(content, style)
        
        # Calculate means
        output_mean = output.mean(dim=[2, 3])
        style_mean = style.mean(dim=[2, 3])
        
        # Should be close (allowing small numerical errors)
        assert torch.allclose(output_mean, style_mean, atol=1e-5)
    
    def test_output_var_matches_style_var(self):
        """Test that output channel variances match style variances."""
        content = torch.randn(1, 512, 14, 14)
        style = torch.randn(1, 512, 14, 14)
        
        output = adain_function(content, style)
        
        # Calculate variances
        output_var = output.var(dim=[2, 3])
        style_var = style.var(dim=[2, 3])
        
        # Should be close
        assert torch.allclose(output_var, style_var, atol=1e-5)
    
    def test_gradient_flows_to_content(self):
        """Test that gradients flow back to content input."""
        content = torch.randn(1, 512, 14, 14, requires_grad=True)
        style = torch.randn(1, 512, 14, 14)
        
        output = adain_function(content, style)
        loss = output.sum()
        loss.backward()
        
        assert content.grad is not None
        assert content.grad.shape == content.shape
    
    def test_works_with_different_batch_sizes(self):
        """Test AdaIN works with various batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            content = torch.randn(batch_size, 512, 14, 14)
            style = torch.randn(batch_size, 512, 14, 14)
            output = adain_function(content, style)
            assert output.shape == content.shape
    
    def test_works_with_different_spatial_dims(self):
        """Test AdaIN works with different spatial dimensions."""
        for h, w in [(7, 7), (14, 14), (28, 28), (56, 56)]:
            content = torch.randn(1, 512, h, w)
            style = torch.randn(1, 512, h, w)
            output = adain_function(content, style)
            assert output.shape == content.shape


class TestAdaINLayer:
    """Tests for AdaIN layer class."""
    
    def test_layer_initialization(self):
        """Test that AdaIN layer initializes correctly."""
        adain = AdaIN()
        assert isinstance(adain, nn.Module)
    
    def test_layer_forward(self):
        """Test AdaIN layer forward pass."""
        adain = AdaIN()
        content = torch.randn(2, 512, 14, 14)
        style = torch.randn(2, 512, 14, 14)
        
        output = adain(content, style)
        
        assert output.shape == content.shape
    
    def test_layer_preserves_device(self):
        """Test that AdaIN preserves device (CPU/CUDA)."""
        adain = AdaIN()
        content = torch.randn(1, 512, 14, 14)
        style = torch.randn(1, 512, 14, 14)
        
        output = adain(content, style)
        
        assert output.device == content.device
    
    def test_layer_with_eps(self):
        """Test that epsilon prevents division by zero."""
        adain = AdaIN(eps=1e-5)
        
        # Create content with zero variance
        content = torch.ones(1, 512, 14, 14)
        style = torch.randn(1, 512, 14, 14)
        
        # Should not raise error
        output = adain(content, style)
        assert output.shape == content.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestAdaINEdgeCases:
    """Edge case tests for AdaIN."""
    
    def test_constant_content(self):
        """Test AdaIN with constant content image."""
        content = torch.ones(1, 512, 14, 14) * 0.5
        style = torch.randn(1, 512, 14, 14)
        
        output = adain_function(content, style)
        
        assert not torch.isnan(output).any()
        assert output.shape == content.shape
    
    def test_constant_style(self):
        """Test AdaIN with constant style image."""
        content = torch.randn(1, 512, 14, 14)
        style = torch.ones(1, 512, 14, 14) * 0.5
        
        output = adain_function(content, style)
        
        # Output should have mean close to 0.5 and very small variance
        output_mean = output.mean(dim=[2, 3])
        assert torch.allclose(output_mean, torch.ones_like(output_mean) * 0.5, atol=1e-4)
    
    def test_identical_inputs(self):
        """Test AdaIN when content and style are identical."""
        x = torch.randn(1, 512, 14, 14)
        
        output = adain_function(x, x)
        
        # Output should be normalized version of input
        assert output.shape == x.shape
        # Mean should be close to original mean, variance close to original variance
        output_mean = output.mean(dim=[2, 3])
        x_mean = x.mean(dim=[2, 3])
        assert torch.allclose(output_mean, x_mean, atol=0.1)
    
    def test_large_values(self):
        """Test AdaIN with large values."""
        content = torch.randn(1, 512, 14, 14) * 100
        style = torch.randn(1, 512, 14, 14) * 100
        
        output = adain_function(content, style)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
