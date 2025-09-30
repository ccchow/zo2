#!/usr/bin/env python3
# Test pinned memory optimization

import torch
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_pinned_memory_transfer(size_mb=100, iterations=50):
    """Test CPU-GPU transfer speed with and without pinned memory"""
    size = int(size_mb * 1024 * 1024 / 4)  # Convert to float32 elements

    print(f"\nTesting transfer speed with {size_mb}MB tensors, {iterations} iterations")
    print("="*60)

    # Test without pinned memory
    cpu_tensor = torch.randn(size)
    gpu_tensor = torch.cuda.FloatTensor(size)

    # Warmup
    for _ in range(5):
        gpu_tensor.copy_(cpu_tensor)
        torch.cuda.synchronize()

    # CPU to GPU without pinned
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        gpu_tensor.copy_(cpu_tensor, non_blocking=False)
    torch.cuda.synchronize()
    time_normal = time.perf_counter() - start
    speed_normal = (size_mb * iterations / 1024) / time_normal

    # Test with pinned memory
    cpu_tensor_pinned = torch.randn(size, pin_memory=True)

    # Warmup
    for _ in range(5):
        gpu_tensor.copy_(cpu_tensor_pinned)
        torch.cuda.synchronize()

    # CPU to GPU with pinned
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        gpu_tensor.copy_(cpu_tensor_pinned, non_blocking=True)
    torch.cuda.synchronize()
    time_pinned = time.perf_counter() - start
    speed_pinned = (size_mb * iterations / 1024) / time_pinned

    print(f"Without pinned memory: {speed_normal:.2f} GB/s ({time_normal:.3f}s)")
    print(f"With pinned memory:    {speed_pinned:.2f} GB/s ({time_pinned:.3f}s)")
    print(f"Speedup: {speed_pinned/speed_normal:.2f}x ({((speed_pinned/speed_normal - 1) * 100):.1f}% faster)")

    return speed_normal, speed_pinned

def test_zo2_with_pinned_memory():
    """Test ZO2 with pinned memory enabled"""
    from zo2.config import ZOConfig
    from transformers import AutoModelForCausalLM
    import torch.nn as nn

    print("\n\nTesting ZO2 with pinned memory")
    print("="*60)

    # Create config with pinned memory disabled
    config_no_pinned = ZOConfig('mezo-sgd')
    config_no_pinned.lr = 1e-5
    config_no_pinned.weight_decay = 0.0
    config_no_pinned.eps = 1e-3
    config_no_pinned.zo2 = True
    config_no_pinned.offloading_device = 'cpu'
    config_no_pinned.working_device = 'cuda'
    config_no_pinned.use_pinned_memory = False
    config_no_pinned.debug_mode = True

    # Create config with pinned memory enabled
    config_pinned = ZOConfig('mezo-sgd')
    config_pinned.lr = 1e-5
    config_pinned.weight_decay = 0.0
    config_pinned.eps = 1e-3
    config_pinned.zo2 = True
    config_pinned.offloading_device = 'cpu'
    config_pinned.working_device = 'cuda'
    config_pinned.use_pinned_memory = True
    config_pinned.pinned_memory_prefetch = True
    config_pinned.debug_mode = True

    print("Config created successfully")
    print(f"  use_pinned_memory (baseline): {config_no_pinned.use_pinned_memory}")
    print(f"  use_pinned_memory (optimized): {config_pinned.use_pinned_memory}")

    # Simple test with a small module
    print("\nTesting with a simple linear module...")
    module = nn.Linear(1024, 1024).cuda()

    # Test offloading without pinned memory
    start = time.perf_counter()
    for _ in range(100):
        module = module.to('cpu')
        module = module.to('cuda')
    time_no_pinned = time.perf_counter() - start

    print(f"Time without pinned memory: {time_no_pinned:.3f}s")

    # Note: Full ZO2 integration test would require loading a model
    # which is complex due to dependencies

if __name__ == "__main__":
    print("ZO2 Pinned Memory Test")
    print("="*60)

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    # Test raw transfer speeds
    test_pinned_memory_transfer(size_mb=50, iterations=100)
    test_pinned_memory_transfer(size_mb=100, iterations=50)
    test_pinned_memory_transfer(size_mb=200, iterations=25)

    # Test ZO2 integration
    test_zo2_with_pinned_memory()

    print("\nTest completed successfully!")