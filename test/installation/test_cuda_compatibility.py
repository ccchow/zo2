#!/usr/bin/env python3
"""Test CUDA compatibility and GPU availability."""

import sys
import torch

def main():
    """Test CUDA and GPU compatibility."""
    print("=" * 60)
    print("CUDA Compatibility Test")
    print("=" * 60)
    
    tests_passed = True
    
    # Test 1: PyTorch version
    print("\n1. PyTorch Version:")
    print("-" * 40)
    print(f"PyTorch version: {torch.__version__}")
    
    # Test 2: CUDA availability
    print("\n2. CUDA Availability:")
    print("-" * 40)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("✗ CUDA is not available!")
        tests_passed = False
    else:
        print("✓ CUDA is available")
    
    # Test 3: CUDA version
    print("\n3. CUDA Version:")
    print("-" * 40)
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        
        # Check CUDA version compatibility
        cuda_version = torch.version.cuda
        if cuda_version:
            major_version = float('.'.join(cuda_version.split('.')[:2]))
            if major_version >= 12.1:
                print(f"✓ CUDA version {cuda_version} meets requirement (>= 12.1)")
            else:
                print(f"✗ CUDA version {cuda_version} below requirement (>= 12.1)")
                tests_passed = False
    else:
        print("N/A - CUDA not available")
    
    # Test 4: GPU devices
    print("\n4. GPU Devices:")
    print("-" * 40)
    if cuda_available:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        if num_gpus == 0:
            print("✗ No GPUs detected!")
            tests_passed = False
        else:
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Multi-Processor Count: {props.multi_processor_count}")
    else:
        print("N/A - CUDA not available")
    
    # Test 5: Simple CUDA operation
    print("\n5. CUDA Operations Test:")
    print("-" * 40)
    if cuda_available:
        try:
            # Create tensors on GPU
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            
            # Perform operation
            z = torch.matmul(x, y)
            
            # Check result
            assert z.shape == (100, 100)
            assert z.is_cuda
            
            print("✓ CUDA tensor operations successful")
            
            # Test memory allocation
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  Memory allocated: {mem_allocated:.2f} MB")
            print(f"  Memory reserved: {mem_reserved:.2f} MB")
            
        except Exception as e:
            print(f"✗ CUDA operations failed: {e}")
            tests_passed = False
    else:
        print("Skipped - CUDA not available")
    
    # Test 6: Mixed precision support
    print("\n6. Mixed Precision Support:")
    print("-" * 40)
    if cuda_available and hasattr(torch.cuda, 'amp'):
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            print("✓ Automatic Mixed Precision (AMP) available")
        except Exception as e:
            print(f"✗ AMP not available: {e}")
            tests_passed = False
    else:
        print("N/A - Requires CUDA")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    if tests_passed and cuda_available:
        print("✓ All CUDA compatibility tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed or CUDA not available")
        sys.exit(1)

if __name__ == "__main__":
    main()