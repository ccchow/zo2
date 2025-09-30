#!/usr/bin/env python3
# Copyright (c) 2025 ZO2 Contributors
# Licensed under the Apache License, Version 2.0

"""
Benchmark script for testing pinned memory optimization in ZO2.
This script compares performance with and without pinned memory enabled.
"""

import torch
import torch.nn as nn
import time
import gc
import argparse
from contextlib import contextmanager
from typing import Dict, List, Tuple
import sys
import os

# Add zo2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from zo2.config import ZOConfig
from zo2.model.huggingface.opt import opt_init_fn
from zo2.model.huggingface import zo_hf_init

def get_memory_stats() -> Dict[str, float]:
    """Get current GPU and CPU memory usage."""
    stats = {}
    if torch.cuda.is_available():
        stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

    # Get CPU memory usage
    import psutil
    process = psutil.Process()
    stats['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024

    return stats

@contextmanager
def timer(name: str):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{name}: {end - start:.4f} seconds")

def create_dummy_batch(batch_size: int, seq_length: int, vocab_size: int, device: str):
    """Create dummy input batch for testing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = input_ids.clone()
    return input_ids, labels

def benchmark_transfer_speed(tensor_size_mb: float, use_pinned: bool, num_iterations: int = 100):
    """Benchmark CPU-GPU transfer speed with and without pinned memory."""
    size = int(tensor_size_mb * 1024 * 1024 / 4)  # Convert MB to number of float32 elements

    # Create tensors
    if use_pinned:
        cpu_tensor = torch.randn(size, pin_memory=True)
    else:
        cpu_tensor = torch.randn(size)

    gpu_tensor = torch.cuda.FloatTensor(size)

    # Warm up
    for _ in range(5):
        gpu_tensor.copy_(cpu_tensor)
        torch.cuda.synchronize()

    # CPU to GPU transfer
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        gpu_tensor.copy_(cpu_tensor, non_blocking=use_pinned)
    torch.cuda.synchronize()
    cpu_to_gpu_time = time.perf_counter() - start

    # GPU to CPU transfer
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        cpu_tensor.copy_(gpu_tensor, non_blocking=use_pinned)
    torch.cuda.synchronize()
    gpu_to_cpu_time = time.perf_counter() - start

    return {
        'cpu_to_gpu_gbps': (tensor_size_mb * num_iterations / 1024) / cpu_to_gpu_time,
        'gpu_to_cpu_gbps': (tensor_size_mb * num_iterations / 1024) / gpu_to_cpu_time,
        'cpu_to_gpu_time': cpu_to_gpu_time,
        'gpu_to_cpu_time': gpu_to_cpu_time
    }

def benchmark_model_training(model_name: str, use_pinned: bool, num_steps: int = 10):
    """Benchmark model training with and without pinned memory."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name} with pinned_memory={use_pinned}")
    print(f"{'='*60}")

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Get ZO config
    zo_config = ZOConfig('mezo-sgd')
    zo_config.lr = 1e-5
    zo_config.weight_decay = 0.0
    zo_config.eps = 1e-3
    zo_config.zo2 = True
    zo_config.offloading_device = 'cpu'
    zo_config.working_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    zo_config.overlap = True
    zo_config.use_pinned_memory = use_pinned
    zo_config.pinned_memory_prefetch = use_pinned
    zo_config.debug_mode = True  # For reproducibility

    # Initialize model
    print(f"Initializing model: {model_name}")
    initial_memory = get_memory_stats()

    with timer("Model initialization"):
        with zo_hf_init():
            model = opt_init_fn(model_name, torch.bfloat16, 'cuda')
            model.zo_init(zo_config)

    post_init_memory = get_memory_stats()

    # Create dummy batch
    batch_size = 1
    seq_length = 128
    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 50257

    input_ids, labels = create_dummy_batch(batch_size, seq_length, vocab_size, 'cuda')

    # Training loop
    step_times = []
    print(f"\nRunning {num_steps} training steps...")

    for step in range(num_steps):
        torch.cuda.synchronize()
        step_start = time.perf_counter()

        # Forward pass with ZO
        loss = model.zo_forward(input_ids, labels=labels)

        torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)

        if step % 5 == 0:
            print(f"Step {step}: loss={loss:.4f}, time={step_time:.4f}s")

    # Get final memory stats
    final_memory = get_memory_stats()
    peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times)
    throughput = 1.0 / avg_step_time

    results = {
        'model': model_name,
        'use_pinned_memory': use_pinned,
        'num_steps': num_steps,
        'avg_step_time': avg_step_time,
        'throughput_steps_per_sec': throughput,
        'initial_gpu_mb': initial_memory.get('gpu_allocated_mb', 0),
        'peak_gpu_mb': peak_gpu_memory,
        'final_gpu_mb': final_memory.get('gpu_allocated_mb', 0),
        'initial_cpu_mb': initial_memory.get('cpu_memory_mb', 0),
        'final_cpu_mb': final_memory.get('cpu_memory_mb', 0),
    }

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark pinned memory optimization')
    parser.add_argument('--model', type=str, default='facebook/opt-125m',
                        help='Model to benchmark (e.g., facebook/opt-125m)')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of training steps')
    parser.add_argument('--transfer-test', action='store_true',
                        help='Run transfer speed benchmark')
    parser.add_argument('--transfer-size', type=float, default=100,
                        help='Transfer test tensor size in MB')
    args = parser.parse_args()

    print("ZO2 Pinned Memory Benchmark")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU only.")
    else:
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run transfer speed benchmark if requested
    if args.transfer_test:
        print(f"\nBenchmarking transfer speeds with {args.transfer_size}MB tensors...")

        print("\nWithout pinned memory:")
        results_normal = benchmark_transfer_speed(args.transfer_size, use_pinned=False)
        print(f"  CPU->GPU: {results_normal['cpu_to_gpu_gbps']:.2f} GB/s")
        print(f"  GPU->CPU: {results_normal['gpu_to_cpu_gbps']:.2f} GB/s")

        print("\nWith pinned memory:")
        results_pinned = benchmark_transfer_speed(args.transfer_size, use_pinned=True)
        print(f"  CPU->GPU: {results_pinned['cpu_to_gpu_gbps']:.2f} GB/s")
        print(f"  GPU->CPU: {results_pinned['gpu_to_cpu_gbps']:.2f} GB/s")

        print("\nSpeedup with pinned memory:")
        print(f"  CPU->GPU: {results_pinned['cpu_to_gpu_gbps']/results_normal['cpu_to_gpu_gbps']:.2f}x")
        print(f"  GPU->CPU: {results_pinned['gpu_to_cpu_gbps']/results_normal['gpu_to_cpu_gbps']:.2f}x")

    # Run model training benchmark
    print(f"\nBenchmarking model training: {args.model}")

    # Baseline without pinned memory
    baseline_results = benchmark_model_training(args.model, use_pinned=False, num_steps=args.steps)

    # With pinned memory
    pinned_results = benchmark_model_training(args.model, use_pinned=True, num_steps=args.steps)

    # Print comparison
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nModel: {args.model}")
    print(f"Steps: {args.steps}")

    print("\nWithout Pinned Memory:")
    print(f"  Avg Step Time: {baseline_results['avg_step_time']:.4f}s")
    print(f"  Throughput: {baseline_results['throughput_steps_per_sec']:.2f} steps/s")
    print(f"  Peak GPU Memory: {baseline_results['peak_gpu_mb']:.1f} MB")
    print(f"  CPU Memory: {baseline_results['final_cpu_mb']:.1f} MB")

    print("\nWith Pinned Memory:")
    print(f"  Avg Step Time: {pinned_results['avg_step_time']:.4f}s")
    print(f"  Throughput: {pinned_results['throughput_steps_per_sec']:.2f} steps/s")
    print(f"  Peak GPU Memory: {pinned_results['peak_gpu_mb']:.1f} MB")
    print(f"  CPU Memory: {pinned_results['final_cpu_mb']:.1f} MB")

    print("\nImprovement with Pinned Memory:")
    speedup = baseline_results['avg_step_time'] / pinned_results['avg_step_time']
    print(f"  Speedup: {speedup:.2f}x ({(speedup - 1) * 100:.1f}% faster)")
    print(f"  Throughput Increase: {(pinned_results['throughput_steps_per_sec'] - baseline_results['throughput_steps_per_sec']):.2f} steps/s")

    # Save results to file
    results_file = f"pinned_memory_results_{args.model.replace('/', '_')}.txt"
    with open(results_file, 'w') as f:
        f.write("ZO2 Pinned Memory Benchmark Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Steps: {args.steps}\n\n")
        f.write("Baseline (without pinned memory):\n")
        for key, value in baseline_results.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nWith pinned memory:\n")
        for key, value in pinned_results.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nSpeedup: {speedup:.2f}x\n")

    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()