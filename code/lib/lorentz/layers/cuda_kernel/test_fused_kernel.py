import torch
import time
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    from hyperbolic_conv_python import FusedHyperbolicConv2d
    print("✓ Python wrapper imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Python wrapper: {e}")
    sys.exit(1)

def test_fused_kernel():
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return False
    
    device = torch.device('cuda')
    print(f"✓ Using device: {device}")
    
    # Test parameters
    batch_size = 4
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    height, width = 224, 224
    manifold_k = 1.0
    
    try:
        # Create fused layer
        fused_conv = FusedHyperbolicConv2d(
            manifold_k=manifold_k,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=1,
            bias=True
        ).to(device)
        print("✓ Fused layer created successfully")
        
        # Create test input
        space_part = torch.randn(batch_size, height, width, in_channels - 1, device=device)
        time_part = torch.sqrt(manifold_k + torch.sum(space_part**2, dim=-1, keepdim=True))
        test_input = torch.cat([time_part, space_part], dim=-1)
        print(f"✓ Test input created: {test_input.shape}")
        
        # Forward pass
        output = fused_conv(test_input)
        print(f"✓ Forward pass successful: {output.shape}")
        
        # Verify hyperbolic constraint
        time_component = output[..., 0]
        space_components = output[..., 1:]
        constraint_violation = torch.abs(
            time_component**2 - torch.sum(space_components**2, dim=-1) - manifold_k
        )
        max_violation = constraint_violation.max().item()
        print(f"✓ Max constraint violation: {max_violation:.6f}")
        
        if max_violation > 0.1:
            print(f"⚠ Warning: Large constraint violation detected: {max_violation} > 0.1")
        
        # Simple benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(10):
            _ = fused_conv(test_input)
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10 * 1000
        print(f"✓ Average forward pass time: {avg_time:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Fused Hyperbolic Convolution Kernel...")
    success = test_fused_kernel()
    if success:
        print("\n🎉 All tests passed! The fused kernel is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)