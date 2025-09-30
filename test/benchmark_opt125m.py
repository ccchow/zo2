#!/usr/bin/env python3
"""
Benchmark OPT-125M with and without pinned memory optimization.
"""

import torch
import torch.nn as nn
import time
import gc
import sys
import os
from dataclasses import dataclass
from typing import Dict, List
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from zo2.config import ZOConfig
from zo2.model.huggingface import zo_hf_init
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil

@dataclass
class BenchmarkResult:
    model: str
    use_pinned_memory: bool
    num_steps: int
    avg_step_time: float
    throughput: float
    initial_gpu_mb: float
    peak_gpu_mb: float
    final_gpu_mb: float
    initial_cpu_mb: float
    final_cpu_mb: float
    step_times: List[float]

def get_memory_stats() -> Dict[str, float]:
    """Get current GPU and CPU memory usage."""
    stats = {}
    if torch.cuda.is_available():
        stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

    process = psutil.Process()
    stats['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024

    return stats

def benchmark_opt_125m(use_pinned_memory: bool, num_steps: int = 20, warmup_steps: int = 3):
    """Benchmark OPT-125M with specified configuration."""

    print(f"\n{'='*70}")
    print(f"Benchmarking OPT-125M with pinned_memory={use_pinned_memory}")
    print(f"{'='*70}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Configuration
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
    zo_config.debug_mode = False  # Enable random noise for realistic training

    # Get initial memory
    initial_memory = get_memory_stats()

    # Load model
    print("Loading model...")
    start_load = time.perf_counter()

    with zo_hf_init():
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        model.zo_init(zo_config)

    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time:.2f}s")

    post_init_memory = get_memory_stats()

    # Create dummy batch
    batch_size = 1
    seq_length = 256
    vocab_size = model.config.vocab_size

    # Generate random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device='cuda')
    labels = input_ids.clone()

    # Warmup steps
    print(f"Running {warmup_steps} warmup steps...")
    for i in range(warmup_steps):
        with torch.no_grad():
            loss = model.zo_forward(input_ids, labels=labels)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  Warmup {i+1}/{warmup_steps}: loss={loss.item():.4f}")

    # Benchmark steps
    step_times = []
    losses = []

    print(f"\nRunning {num_steps} benchmark steps...")
    for step in range(num_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_start = time.perf_counter()

        # Forward pass with ZO
        with torch.no_grad():
            loss = model.zo_forward(input_ids, labels=labels)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        losses.append(loss.item())

        # Print progress every 5 steps
        if (step + 1) % 5 == 0:
            avg_time = sum(step_times) / len(step_times)
            print(f"  Step {step+1}/{num_steps}: loss={loss.item():.4f}, "
                  f"time={step_time:.4f}s, avg={avg_time:.4f}s")

    # Get final memory stats
    final_memory = get_memory_stats()
    peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    throughput = 1.0 / avg_step_time

    # Print summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUMMARY - Pinned Memory: {use_pinned_memory}")
    print(f"{'='*70}")
    print(f"Average step time: {avg_step_time:.4f}s (min: {min_step_time:.4f}s, max: {max_step_time:.4f}s)")
    print(f"Throughput: {throughput:.2f} steps/sec")
    print(f"Initial GPU memory: {initial_memory.get('gpu_allocated_mb', 0):.1f} MB")
    print(f"Peak GPU memory: {peak_gpu_memory:.1f} MB")
    print(f"Final GPU memory: {final_memory.get('gpu_allocated_mb', 0):.1f} MB")
    print(f"Initial CPU memory: {initial_memory.get('cpu_memory_mb', 0):.1f} MB")
    print(f"Final CPU memory: {final_memory.get('cpu_memory_mb', 0):.1f} MB")
    print(f"Average loss: {sum(losses)/len(losses):.4f}")

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return BenchmarkResult(
        model="facebook/opt-125m",
        use_pinned_memory=use_pinned_memory,
        num_steps=num_steps,
        avg_step_time=avg_step_time,
        throughput=throughput,
        initial_gpu_mb=initial_memory.get('gpu_allocated_mb', 0),
        peak_gpu_mb=peak_gpu_memory,
        final_gpu_mb=final_memory.get('gpu_allocated_mb', 0),
        initial_cpu_mb=initial_memory.get('cpu_memory_mb', 0),
        final_cpu_mb=final_memory.get('cpu_memory_mb', 0),
        step_times=step_times
    )

def main():
    print("="*70)
    print("OPT-125M Pinned Memory Benchmark")
    print("="*70)

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        sys.exit(1)

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Run benchmarks
    num_steps = 30  # More steps for stable average

    # Baseline without pinned memory
    print("\n" + "="*70)
    print("BASELINE: Without Pinned Memory")
    print("="*70)
    baseline_result = benchmark_opt_125m(use_pinned_memory=False, num_steps=num_steps)

    # Wait a bit between runs
    time.sleep(5)

    # With pinned memory
    print("\n" + "="*70)
    print("OPTIMIZED: With Pinned Memory")
    print("="*70)
    pinned_result = benchmark_opt_125m(use_pinned_memory=True, num_steps=num_steps)

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    speedup = baseline_result.avg_step_time / pinned_result.avg_step_time
    throughput_gain = pinned_result.throughput - baseline_result.throughput

    print(f"\nWithout Pinned Memory:")
    print(f"  Average step time: {baseline_result.avg_step_time:.4f}s")
    print(f"  Throughput: {baseline_result.throughput:.2f} steps/s")
    print(f"  Peak GPU memory: {baseline_result.peak_gpu_mb:.1f} MB")
    print(f"  Final CPU memory: {baseline_result.final_cpu_mb:.1f} MB")

    print(f"\nWith Pinned Memory:")
    print(f"  Average step time: {pinned_result.avg_step_time:.4f}s")
    print(f"  Throughput: {pinned_result.throughput:.2f} steps/s")
    print(f"  Peak GPU memory: {pinned_result.peak_gpu_mb:.1f} MB")
    print(f"  Final CPU memory: {pinned_result.final_cpu_mb:.1f} MB")

    print(f"\nIMPROVEMENT:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Performance gain: {(speedup - 1) * 100:.1f}%")
    print(f"  Throughput increase: {throughput_gain:.2f} steps/s")
    print(f"  Throughput gain: {(pinned_result.throughput / baseline_result.throughput - 1) * 100:.1f}%")

    # Save results to JSON
    results = {
        "baseline": {
            "use_pinned_memory": baseline_result.use_pinned_memory,
            "avg_step_time": baseline_result.avg_step_time,
            "throughput": baseline_result.throughput,
            "peak_gpu_mb": baseline_result.peak_gpu_mb,
            "final_cpu_mb": baseline_result.final_cpu_mb,
        },
        "optimized": {
            "use_pinned_memory": pinned_result.use_pinned_memory,
            "avg_step_time": pinned_result.avg_step_time,
            "throughput": pinned_result.throughput,
            "peak_gpu_mb": pinned_result.peak_gpu_mb,
            "final_cpu_mb": pinned_result.final_cpu_mb,
        },
        "improvement": {
            "speedup": speedup,
            "performance_gain_percent": (speedup - 1) * 100,
            "throughput_increase": throughput_gain,
            "throughput_gain_percent": (pinned_result.throughput / baseline_result.throughput - 1) * 100
        }
    }

    with open("opt125m_pinned_memory_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to opt125m_pinned_memory_results.json")

    return results

if __name__ == "__main__":
    results = main()