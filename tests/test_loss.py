"""Tests for style transfer loss functions."""

import pytest
import torch
import torch.nn as nn

from style_shift.models.loss import ContentLoss, StyleLoss, TVLoss, combine_losses


class TestContentLoss:
    """Tests for ContentLoss."""
    
    def test_content_loss_initializes(self):
        """Test ContentLoss initializes correctly."""
        loss_fn = ContentLoss()
        assert isinstance(loss_fn, nn.Module)
    
    def test_content_loss_returns_scalar(self):
        """Test that content loss returns scalar tensor."""
        loss_fn = ContentLoss()
        gen = torch.randn(1, 512, 14, 14)
        content = torch.randn(1, 512, 14, 14)
        
        loss = loss_fn(gen, content)
        
        assert loss.dim() == 0  # Scalar
    
    def test_content_loss_zero_for_identical(self):
        """Test that identical inputs give zero loss."""
        loss_fn = ContentLoss()
        x = torch.randn(1, 512, 14, 14)
        
        loss = loss_fn(x, x)
        
        assert torch.abs(loss).item() < 1e-6
    
    def test_content_loss_increases_with_difference(self):
        """Test that loss increases with feature difference."""
        loss_fn = ContentLoss()
        content = torch.ones(1, 512, 14, 14)
        
        gen1 = content + 0.1
        gen2 = content + 1.0
        
        loss1 = loss_fn(gen1, content)
        loss2 = loss_fn(gen2, content)
        
        assert loss2 > loss1
    
    def test_content_loss_weight(self):
        """Test that weight parameter scales loss."""
        loss_fn1 = ContentLoss(weight=1.0)
        loss_fn2 = ContentLoss(weight=2.0)
        
        gen = torch.randn(1, 512, 14, 14)
        content = torch.randn(1, 512, 14, 14)
        
        loss1 = loss_fn1(gen, content)
        loss2 = loss_fn2(gen, content)
        
        assert torch.allclose(loss2, loss1 * 2.0)
    
    def test_content_loss_gradient(self):
        """Test that gradients flow through content loss."""
        loss_fn = ContentLoss()
        gen = torch.randn(1, 512, 14, 14, requires_grad=True)
        content = torch.randn(1, 512, 14, 14)
        
        loss = loss_fn(gen, content)
        loss.backward()
        
        assert gen.grad is not None


class TestStyleLoss:
    """Tests for StyleLoss."""
    
    def test_style_loss_initializes(self):
        """Test StyleLoss initializes correctly."""
        loss_fn = StyleLoss()
        assert isinstance(loss_fn, nn.Module)
    
    def test_style_loss_returns_scalar(self):
        """Test that style loss returns scalar tensor."""
        loss_fn = StyleLoss()
        gen = torch.randn(1, 512, 28, 28)
        style = torch.randn(1, 512, 28, 28)
        
        loss = loss_fn(gen, style)
        
        assert loss.dim() == 0
    
    def test_style_loss_zero_for_identical(self):
        """Test that identical inputs give zero style loss."""
        loss_fn = StyleLoss()
        x = torch.randn(1, 512, 28, 28)
        
        loss = loss_fn(x, x)
        
        assert torch.abs(loss).item() < 1e-6
    
    def test_style_loss_with_different_sizes(self):
        """Test style loss works with different feature sizes."""
        loss_fn = StyleLoss()
        
        for size in [14, 28, 56]:
            gen = torch.randn(1, 512, size, size)
            style = torch.randn(1, 512, size, size)
            
            loss = loss_fn(gen, style)
            assert loss.dim() == 0
    
    def test_style_loss_weight(self):
        """Test that weight parameter scales style loss."""
        loss_fn1 = StyleLoss(weight=1.0)
        loss_fn2 = StyleLoss(weight=2.0)
        
        gen = torch.randn(1, 512, 28, 28)
        style = torch.randn(1, 512, 28, 28)
        
        loss1 = loss_fn1(gen, style)
        loss2 = loss_fn2(gen, style)
        
        assert torch.allclose(loss2, loss1 * 2.0)
    
    def test_gram_matrix_computation(self):
        """Test internal Gram matrix computation."""
        loss_fn = StyleLoss()
        x = torch.randn(1, 3, 4, 4)
        
        gram = loss_fn._gram_matrix(x)
        
        # Gram matrix should be (N, C, C)
        assert gram.shape == (1, 3, 3)
        
        # Gram matrix should be symmetric
        assert torch.allclose(gram, gram.transpose(1, 2), atol=1e-6)


class TestTVLoss:
    """Tests for TVLoss (Total Variation Loss)."""
    
    def test_tv_loss_initializes(self):
        """Test TVLoss initializes correctly."""
        loss_fn = TVLoss()
        assert isinstance(loss_fn, nn.Module)
    
    def test_tv_loss_returns_scalar(self):
        """Test that TV loss returns scalar tensor."""
        loss_fn = TVLoss()
        x = torch.randn(1, 3, 224, 224)
        
        loss = loss_fn(x)
        
        assert loss.dim() == 0
    
    def test_tv_loss_zero_for_constant(self):
        """Test that constant image gives zero TV loss."""
        loss_fn = TVLoss()
        x = torch.ones(1, 3, 224, 224) * 0.5
        
        loss = loss_fn(x)
        
        assert torch.abs(loss).item() < 1e-6
    
    def test_tv_loss_increases_with_noise(self):
        """Test that noisy images have higher TV loss."""
        loss_fn = TVLoss()
        
        smooth = torch.ones(1, 3, 64, 64) * 0.5
        noisy = smooth + torch.randn(1, 3, 64, 64) * 0.1
        
        loss_smooth = loss_fn(smooth)
        loss_noisy = loss_fn(noisy)
        
        assert loss_noisy > loss_smooth
    
    def test_tv_loss_weight(self):
        """Test that weight parameter scales TV loss."""
        loss_fn1 = TVLoss(weight=1.0)
        loss_fn2 = TVLoss(weight=2.0)
        
        x = torch.randn(1, 3, 64, 64)
        
        loss1 = loss_fn1(x)
        loss2 = loss_fn2(x)
        
        assert torch.allclose(loss2, loss1 * 2.0)


class TestCombineLosses:
    """Tests for combine_losses function."""
    
    def test_combine_returns_scalar(self):
        """Test that combined loss returns scalar."""
        c_loss = torch.tensor(1.0)
        s_loss = torch.tensor(2.0)
        
        total = combine_losses(c_loss, s_loss)
        
        assert total.dim() == 0
    
    def test_combine_with_alpha_beta(self):
        """Test combining with content and style weights."""
        c_loss = torch.tensor(1.0)
        s_loss = torch.tensor(2.0)
        
        # alpha=1.0, beta=0.1
        total = combine_losses(c_loss, s_loss, alpha=1.0, beta=0.1)
        
        expected = 1.0 * 1.0 + 0.1 * 2.0
        assert torch.allclose(total, torch.tensor(expected))
    
    def test_combine_with_tv_loss(self):
        """Test combining with TV loss."""
        c_loss = torch.tensor(1.0)
        s_loss = torch.tensor(2.0)
        tv_loss = torch.tensor(0.5)
        
        total = combine_losses(c_loss, s_loss, tv_loss, alpha=1.0, beta=1.0, gamma=0.001)
        
        expected = 1.0 + 2.0 + 0.001 * 0.5
        assert torch.allclose(total, torch.tensor(expected))
    
    def test_combine_without_tv_loss(self):
        """Test combining without TV loss."""
        c_loss = torch.tensor(1.0)
        s_loss = torch.tensor(2.0)
        
        total = combine_losses(c_loss, s_loss, tv_loss=None)
        
        assert total.dim() == 0
