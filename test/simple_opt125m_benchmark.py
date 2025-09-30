#!/usr/bin/env python3
"""
Simple benchmark for OPT-125M with/without pinned memory.
"""

import torch
import time
import gc
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from zo2.config import ZOConfig
from zo2.model.huggingface.zo_init import zo_hf_init
from transformers import AutoTokenizer
import psutil

def get_memory_mb():
    """Get memory usage in MB."""
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        gpu_mb = 0
    cpu_mb = psutil.Process().memory_info().rss / 1024 / 1024
    return gpu_mb, cpu_mb

def run_benchmark(use_pinned_memory: bool, num_steps: int = 20):
    """Run benchmark with specified configuration."""
    print(f"\n{'='*60}")
    print(f"Testing OPT-125M with pinned_memory={use_pinned_memory}")
    print(f"{'='*60}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Create config
    zo_config = ZOConfig('mezo-sgd')
    zo_config.lr = 1e-5
    zo_config.weight_decay = 0.01
    zo_config.eps = 1e-3
    zo_config.zo2 = True
    zo_config.offloading_device = 'cpu'
    zo_config.working_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    zo_config.overlap = True
    zo_config.use_pinned_memory = use_pinned_memory
    zo_config.pinned_memory_prefetch = use_pinned_memory

    # Load model
    print("Loading model...")
    with zo_hf_init(zo_config):
        from transformers import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torch_dtype=torch.bfloat16
        )
        model.zo_init(zo_config)

    # Move to device
    if torch.cuda.is_available():
        model = model.cuda()

    # Create dummy input
    batch_size = 1
    seq_length = 256
    vocab_size = 50272  # OPT vocab size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    labels = input_ids.clone()

    # Warmup
    print("Warming up...")
    for _ in range(3):
        loss = model.zo_forward(input_ids, labels=labels)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    print(f"Running {num_steps} steps...")
    step_times = []

    for step in range(num_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        loss = model.zo_forward(input_ids, labels=labels)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_time = time.perf_counter() - start_time
        step_times.append(step_time)

        if (step + 1) % 5 == 0:
            avg_time = sum(step_times) / len(step_times)
            print(f"  Step {step+1}/{num_steps}: {step_time:.4f}s (avg: {avg_time:.4f}s)")

    # Get final stats
    avg_step_time = sum(step_times) / len(step_times)
    throughput = 1.0 / avg_step_time

    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_gpu_mb = 0

    final_gpu_mb, final_cpu_mb = get_memory_mb()

    print(f"\nResults:")
    print(f"  Avg step time: {avg_step_time:.4f}s")
    print(f"  Throughput: {throughput:.2f} steps/s")
    print(f"  Peak GPU: {peak_gpu_mb:.1f} MB")
    print(f"  Final CPU: {final_cpu_mb:.1f} MB")

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        'avg_step_time': avg_step_time,
        'throughput': throughput,
        'peak_gpu_mb': peak_gpu_mb,
        'final_cpu_mb': final_cpu_mb
    }

def main():
    print("OPT-125M Pinned Memory Benchmark")
    print("="*60)

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    # Run benchmarks
    baseline = run_benchmark(use_pinned_memory=False, num_steps=20)
    time.sleep(3)  # Wait between runs
    optimized = run_benchmark(use_pinned_memory=True, num_steps=20)

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    speedup = baseline['avg_step_time'] / optimized['avg_step_time']

    print("\nWithout Pinned Memory:")
    print(f"  Step time: {baseline['avg_step_time']:.4f}s")
    print(f"  Throughput: {baseline['throughput']:.2f} steps/s")
    print(f"  Peak GPU: {baseline['peak_gpu_mb']:.1f} MB")

    print("\nWith Pinned Memory:")
    print(f"  Step time: {optimized['avg_step_time']:.4f}s")
    print(f"  Throughput: {optimized['throughput']:.2f} steps/s")
    print(f"  Peak GPU: {optimized['peak_gpu_mb']:.1f} MB")

    print(f"\nSpeedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    print(f"Throughput gain: {optimized['throughput'] - baseline['throughput']:.2f} steps/s")

if __name__ == "__main__":
    main()