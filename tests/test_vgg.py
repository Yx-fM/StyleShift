"""Tests for VGG-19 encoder."""

import pytest
import torch
import torch.nn as nn

from style_shift.models.vgg import VGG19Encoder, get_vgg19, VGG_LAYERS


class TestVGG19EncoderInit:
    """Tests for VGG19Encoder initialization."""
    
    def test_encoder_initializes(self):
        """Test that VGG19Encoder initializes without error."""
        encoder = VGG19Encoder(pretrained=False)
        assert isinstance(encoder, nn.Module)
    
    def test_encoder_has_vgg(self):
        """Test that encoder has vgg attribute."""
        encoder = VGG19Encoder(pretrained=False)
        assert hasattr(encoder, 'vgg')
        assert encoder.vgg is not None
    
    def test_encoder_requires_grad_false(self):
        """Test that encoder parameters are frozen by default."""
        encoder = VGG19Encoder(pretrained=False)
        for param in encoder.parameters():
            assert not param.requires_grad


class TestVGG19EncoderForward:
    """Tests for VGG19Encoder forward pass."""
    
    def test_content_feature_shape(self):
        """Test that content features have correct shape."""
        encoder = VGG19Encoder()
        x = torch.randn(1, 3, 224, 224)
        
        content_feat, style_feats = encoder(x)
        
        # Content feature from conv4_2 should be (N, 512, 28, 28)
        assert content_feat.shape == (1, 512, 28, 28)
    
    def test_style_feature_shapes(self):
        """Test that style features have correct shapes for all layers."""
        encoder = VGG19Encoder()
        x = torch.randn(1, 3, 224, 224)
        
        content_feat, style_feats = encoder(x)
        
        # Check all style layers exist
        expected_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        for layer in expected_layers:
            assert layer in style_feats
        
        # Check shapes
        assert style_feats['conv1_1'].shape == (1, 64, 224, 224)
        assert style_feats['conv2_1'].shape == (1, 128, 112, 112)
        assert style_feats['conv3_1'].shape == (1, 256, 56, 56)
        assert style_feats['conv4_1'].shape == (1, 512, 28, 28)
        assert style_feats['conv5_1'].shape == (1, 512, 14, 14)
    
    def test_batch_processing(self):
        """Test encoder works with different batch sizes."""
        encoder = VGG19Encoder()
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            content_feat, style_feats = encoder(x)
            
            assert content_feat.shape[0] == batch_size
            for layer_feat in style_feats.values():
                assert layer_feat.shape[0] == batch_size
    
    def test_different_input_sizes(self):
        """Test encoder works with different input sizes."""
        encoder = VGG19Encoder()
        
        for size in [128, 224, 256, 512]:
            x = torch.randn(1, 3, size, size)
            content_feat, style_feats = encoder(x)
            
            # Content feature should be 1/8 of input size (conv4_2)
            assert content_feat.shape[2:] == (size // 8, size // 8)
    
    def test_gradient_flow(self):
        """Test that gradients flow through encoder."""
        encoder = VGG19Encoder()
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        content_feat, style_feats = encoder(x)
        loss = content_feat.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestGetVgg19:
    """Tests for get_vgg19 utility function."""
    
    def test_get_vgg19_returns_sequential(self):
        """Test that get_vgg19 returns nn.Sequential."""
        vgg = get_vgg19()
        assert isinstance(vgg, nn.Sequential)
    
    def test_get_vgg19_has_correct_layers(self):
        """Test that VGG-19 has correct number of layers."""
        vgg = get_vgg19()
        # VGG-19 should have 37 layers (including ReLU and MaxPool)
        assert len(vgg) == 37
    
    def test_get_vgg19_pretrained(self):
        """Test that pretrained weights are loaded."""
        vgg = get_vgg19(pretrained=True)
        
        # Check that parameters are not all zeros
        for param in vgg.parameters():
            assert param.abs().sum() > 0


class TestVggLayers:
    """Tests for VGG layer configuration."""
    
    def test_vgg_layers_constant_exists(self):
        """Test that VGG_LAYERS constant is defined."""
        assert isinstance(VGG_LAYERS, dict)
        assert 'content' in VGG_LAYERS
        assert 'style' in VGG_LAYERS
    
    def test_content_layer_index(self):
        """Test that content layer index is correct."""
        assert VGG_LAYERS['content'] == 23  # conv4_2
    
    def test_style_layer_indices(self):
        """Test that style layer indices are correct."""
        style_indices = VGG_LAYERS['style']
        assert len(style_indices) == 5
        # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
        assert 0 in style_indices  # conv1_1
        assert 5 in style_indices  # conv2_1
        assert 10 in style_indices  # conv3_1
        assert 19 in style_indices  # conv4_1
        assert 28 in style_indices  # conv5_1
