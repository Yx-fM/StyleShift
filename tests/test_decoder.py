"""Tests for Decoder network."""

import pytest
import torch
import torch.nn as nn

from style_shift.models.decoder import Decoder


class TestDecoderInit:
    """Tests for Decoder initialization."""
    
    def test_decoder_initializes(self):
        """Test that Decoder initializes without error."""
        decoder = Decoder()
        assert isinstance(decoder, nn.Module)
    
    def test_decoder_has_layers(self):
        """Test that decoder has layers attribute."""
        decoder = Decoder()
        assert hasattr(decoder, 'layers')
        assert decoder.layers is not None
    
    def test_decoder_parameter_count(self):
        """Test decoder has reasonable number of parameters."""
        decoder = Decoder()
        num_params = sum(p.numel() for p in decoder.parameters())
        # Should be in millions range
        assert num_params > 1_000_000


class TestDecoderForward:
    """Tests for Decoder forward pass."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        decoder = Decoder()
        x = torch.randn(1, 512, 14, 14)
        
        output = decoder(x)
        
        # Should reconstruct to 16x spatial size
        assert output.shape == (1, 3, 224, 224)
    
    def test_batch_processing(self):
        """Test decoder works with different batch sizes."""
        decoder = Decoder()
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 512, 14, 14)
            output = decoder(x)
            
            assert output.shape[0] == batch_size
            assert output.shape[1:] == (3, 224, 224)
    
    def test_different_feature_sizes(self):
        """Test decoder works with different input feature sizes."""
        decoder = Decoder()
        
        # Input should be (N, 512, H, W) where H,W >= 14
        for h, w in [(14, 14), (28, 28), (56, 56)]:
            x = torch.randn(1, 512, h, w)
            output = decoder(x)
            
            # Output should be 16x input size
            expected_h = h * 16
            expected_w = w * 16
            assert output.shape == (1, 3, expected_h, expected_w)
    
    def test_output_range(self):
        """Test that output values are in reasonable range."""
        decoder = Decoder()
        x = torch.randn(1, 512, 14, 14)
        
        output = decoder(x)
        
        # Output should be finite (no NaN or Inf)
        assert torch.isfinite(output).all()
        
        # Note: Without final activation, output can be outside [0, 1]
        # Image I/O will handle clamping
    
    def test_gradient_flow(self):
        """Test that gradients flow through decoder."""
        decoder = Decoder()
        x = torch.randn(1, 512, 14, 14, requires_grad=True)
        
        output = decoder(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestDecoderArchitecture:
    """Tests for Decoder architecture details."""
    
    def test_uses_upsampling(self):
        """Test that decoder uses upsampling layers."""
        decoder = Decoder()
        
        # Count upsampling layers
        upsample_count = sum(
            1 for module in decoder.modules()
            if isinstance(module, nn.Upsample)
        )
        
        # Should have 4 upsampling layers (512->256->128->64->3)
        assert upsample_count >= 4
    
    def test_uses_conv_layers(self):
        """Test that decoder uses convolutional layers."""
        decoder = Decoder()
        
        # Count conv layers
        conv_count = sum(
            1 for module in decoder.modules()
            if isinstance(module, nn.Conv2d)
        )
        
        # Should have multiple conv layers (actual: 9)
        assert conv_count >= 9
    
    def test_uses_relu(self):
        """Test that decoder uses ReLU activations."""
        decoder = Decoder()
        
        # Count ReLU layers
        relu_count = sum(
            1 for module in decoder.modules()
            if isinstance(module, nn.ReLU)
        )
        
        # Should have multiple ReLU layers (actual: 8)
        assert relu_count >= 8
