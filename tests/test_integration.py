#!/usr/bin/env python3
"""End-to-end MVP integration test for StyleShift."""

import sys
import time
from pathlib import Path
from PIL import Image
import torch


def test_full_pipeline():
    """Test complete style transfer pipeline."""
    from style_shift.core.style_transfer import StyleTransfer, StyleTransferConfig
    
    print("=" * 60)
    print("🧪 Running StyleShift Integration Test...")
    print("=" * 60)
    
    # 1. Create test images
    content = Image.new('RGB', (512, 512), color='red')
    style = Image.new('RGB', (512, 512), color='blue')
    
    # 2. Initialize StyleTransfer
    config = StyleTransferConfig(max_size=512, device='cpu')
    st = StyleTransfer(config)
    
    # 3. Perform style transfer with timing
    print("\n⏱️  Starting style transfer (512×512, CPU)...")
    start = time.time()
    result = st.transfer(content=content, style=style, alpha=0.8)
    elapsed = time.time() - start
    
    # 4. Validate output
    assert isinstance(result, Image.Image), "Result should be PIL Image"
    assert result.mode == 'RGB', f"Result should be RGB, got {result.mode}"
    assert result.size == (512, 512), f"Expected (512, 512), got {result.size}"
    
    print(f"✓ Output validation passed")
    print(f"  - Type: {type(result).__name__}")
    print(f"  - Mode: {result.mode}")
    print(f"  - Size: {result.size}")
    
    # 5. Performance validation
    print(f"\n⏱️  Inference time: {elapsed:.2f}s")
    assert elapsed < 5.0, f"CPU inference should be < 5s, got {elapsed:.2f}s"
    print(f"✓ Performance test passed (< 5s)")
    
    # 6. Test with different alpha values
    print("\n🎨 Testing alpha interpolation...")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = st.transfer(content=content, style=style, alpha=alpha)
        assert result is not None
        print(f"  α={alpha:.2f}: ✓")
    
    # 7. Test tensor input
    print("\n🔧 Testing tensor input...")
    content_tensor = torch.rand(1, 3, 512, 512)
    result = st.transfer(content=content_tensor, style=style, alpha=0.8)
    assert result is not None
    print("  Tensor input: ✓")
    
    print("\n" + "=" * 60)
    print("✅ Integration test PASSED")
    print("=" * 60)
    
    return True


def test_preprocess_postprocess():
    """Test preprocessing and postprocessing functions."""
    from style_shift.core import preprocess, postprocess
    
    print("\n🔧 Testing preprocessing/postprocessing...")
    
    # Test resize
    img = Image.new('RGB', (1024, 768))
    resized = preprocess.resize_image(img, 512)
    assert resized.size == (512, 384), f"Expected (512, 384), got {resized.size}"
    print("  Resize: ✓")
    
    # Test normalize/denormalize
    tensor = torch.rand(3, 224, 224)
    normalized = preprocess.normalize(tensor)
    denormalized = postprocess.denormalize(normalized)
    diff = (tensor - denormalized).abs().max().item()
    assert diff < 1e-5, f"Roundtrip diff too large: {diff}"
    print(f"  Normalize roundtrip: ✓ (diff={diff:.2e})")
    
    # Test preprocess_image
    tensor = preprocess.preprocess_image(img, max_size=256)
    assert tensor.dim() == 3
    print("  preprocess_image: ✓")
    
    # Test postprocess_image
    pil = postprocess.postprocess_image(tensor)
    assert isinstance(pil, Image.Image)
    print("  postprocess_image: ✓")
    
    print("  All pre/post tests: ✓")
    
    return True


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("🎨 StyleShift MVP Integration Tests")
    print("=" * 60)
    
    try:
        # Test 1: Pre/Post processing
        test_preprocess_postprocess()
        
        # Test 2: Full pipeline
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)
        print("\nWave 3 (Core Business Layer): COMPLETE ✓")
        print("Performance: < 5s for 512×512 on CPU ✓")
        print("\nNext steps:")
        print("  - Wave 5: CLI interface")
        print("  - Wave 5: Web UI (Gradio)")
        print("  - Wave 6: Final testing & documentation")
        
        return 0
    
    except AssertionError as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
