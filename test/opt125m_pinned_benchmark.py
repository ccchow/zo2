#!/usr/bin/env python3
"""
Benchmark OPT-125M with and without pinned memory using existing zo2 structure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
import gc
from zo2.config.mezo_sgd import MeZOSGDConfig
from zo2.model.huggingface.opt.mezo_sgd import zo2
from transformers import OPTConfig
import psutil

def get_memory_stats():
    """Get current memory usage."""
    stats = {}
    if torch.cuda.is_available():
        stats['gpu_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        stats['gpu_peak_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        stats['gpu_mb'] = 0
        stats['gpu_peak_mb'] = 0

    process = psutil.Process()
    stats['cpu_mb'] = process.memory_info().rss / 1024 / 1024
    return stats

def run_benchmark(use_pinned_memory: bool, num_steps: int = 20):
    """Run benchmark with specified configuration."""

    print(f"\n{'='*70}")
    print(f"Benchmarking OPT-125M with pinned_memory={use_pinned_memory}")
    print(f"{'='*70}")

    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Create ZO config
    zo_config = MeZOSGDConfig()
    zo_config.lr = 1e-5
    zo_config.weight_decay = 0.01
    zo_config.eps = 1e-3
    zo_config.zo2 = True
    zo_config.offloading_device = 'cpu'
    zo_config.working_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    zo_config.overlap = True
    zo_config.use_pinned_memory = use_pinned_memory
    zo_config.pinned_memory_prefetch = use_pinned_memory
    zo_config.debug_mode = False  # Use real training

    # Create OPT-125M config
    model_config = OPTConfig(
        vocab_size=50272,
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        num_attention_heads=12,
        max_position_embeddings=2048,
        dropout=0.1,
        attention_dropout=0.0,
        activation_function="relu",
        init_std=0.02,
        layerdrop=0.0,
        use_cache=False,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        enable_bias=True,
        layer_norm_elementwise_affine=True,
        _name_or_path="facebook/opt-125m"
    )

    # Get initial memory
    initial_mem = get_memory_stats()
    print(f"Initial memory - GPU: {initial_mem['gpu_mb']:.1f}MB, CPU: {initial_mem['cpu_mb']:.1f}MB")

    # Create model
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = zo2.OPTForCausalLM(model_config)
    model.zo_init(zo_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.1f}M")

    # Get post-load memory
    post_load_mem = get_memory_stats()
    print(f"After loading - GPU: {post_load_mem['gpu_mb']:.1f}MB, CPU: {post_load_mem['cpu_mb']:.1f}MB")

    # Prepare dummy data
    batch_size = 1
    seq_length = 256
    vocab_size = model_config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = input_ids.clone()

    # Warmup
    print("Warming up (3 steps)...")
    model.zo_train()
    for i in range(3):
        outputs = model(input_ids=input_ids, labels=labels)
        # Handle different output types
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        else:
            loss = outputs[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  Warmup {i+1}: loss={loss.item():.4f}")

    # Benchmark
    print(f"\nRunning {num_steps} benchmark steps...")
    step_times = []
    losses = []

    for step in range(num_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        # Handle different output types
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        else:
            loss = outputs[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        step_time = time.perf_counter() - start_time
        step_times.append(step_time)
        losses.append(loss.item())

        # Progress update
        if (step + 1) % 5 == 0:
            avg_time = sum(step_times) / len(step_times)
            print(f"  Step {step+1}/{num_steps}: time={step_time:.4f}s, avg={avg_time:.4f}s, loss={loss.item():.4f}")

    # Get final memory stats
    final_mem = get_memory_stats()

    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    throughput = 1.0 / avg_step_time
    avg_loss = sum(losses) / len(losses)

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS - Pinned Memory: {use_pinned_memory}")
    print(f"{'='*70}")
    print(f"Timing:")
    print(f"  Average step: {avg_step_time:.4f}s")
    print(f"  Min step: {min_step_time:.4f}s")
    print(f"  Max step: {max_step_time:.4f}s")
    print(f"  Throughput: {throughput:.2f} steps/sec")
    print(f"Memory:")
    print(f"  Peak GPU: {final_mem['gpu_peak_mb']:.1f} MB")
    print(f"  Final GPU: {final_mem['gpu_mb']:.1f} MB")
    print(f"  Final CPU: {final_mem['cpu_mb']:.1f} MB")
    print(f"Training:")
    print(f"  Average loss: {avg_loss:.4f}")

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        'avg_step_time': avg_step_time,
        'min_step_time': min_step_time,
        'max_step_time': max_step_time,
        'throughput': throughput,
        'peak_gpu_mb': final_mem['gpu_peak_mb'],
        'final_gpu_mb': final_mem['gpu_mb'],
        'final_cpu_mb': final_mem['cpu_mb'],
        'avg_loss': avg_loss
    }

def main():
    print("="*70)
    print("OPT-125M Pinned Memory Comparison Benchmark")
    print("="*70)

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Number of steps for stable measurement
    num_steps = 30

    # Run baseline (without pinned memory)
    print("\n" + "="*70)
    print("PHASE 1: BASELINE (Without Pinned Memory)")
    print("="*70)
    baseline = run_benchmark(use_pinned_memory=False, num_steps=num_steps)

    # Wait between runs
    print("\nWaiting 5 seconds before next run...")
    time.sleep(5)

    # Run optimized (with pinned memory)
    print("\n" + "="*70)
    print("PHASE 2: OPTIMIZED (With Pinned Memory)")
    print("="*70)
    optimized = run_benchmark(use_pinned_memory=True, num_steps=num_steps)

    # Calculate improvements
    speedup = baseline['avg_step_time'] / optimized['avg_step_time']
    throughput_gain = optimized['throughput'] - baseline['throughput']
    throughput_gain_pct = (optimized['throughput'] / baseline['throughput'] - 1) * 100

    # Print comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    print("\n1. WITHOUT Pinned Memory:")
    print(f"   - Average step time: {baseline['avg_step_time']:.4f}s")
    print(f"   - Throughput: {baseline['throughput']:.2f} steps/s")
    print(f"   - Peak GPU memory: {baseline['peak_gpu_mb']:.1f} MB")
    print(f"   - CPU memory: {baseline['final_cpu_mb']:.1f} MB")

    print("\n2. WITH Pinned Memory:")
    print(f"   - Average step time: {optimized['avg_step_time']:.4f}s")
    print(f"   - Throughput: {optimized['throughput']:.2f} steps/s")
    print(f"   - Peak GPU memory: {optimized['peak_gpu_mb']:.1f} MB")
    print(f"   - CPU memory: {optimized['final_cpu_mb']:.1f} MB")

    print("\n3. IMPROVEMENT:")
    print(f"   - Speedup: {speedup:.2f}x")
    print(f"   - Performance gain: {(speedup - 1) * 100:.1f}%")
    print(f"   - Throughput increase: +{throughput_gain:.2f} steps/s")
    print(f"   - Throughput gain: +{throughput_gain_pct:.1f}%")

    # Save results
    import json
    results = {
        'model': 'facebook/opt-125m',
        'num_steps': num_steps,
        'baseline': baseline,
        'optimized': optimized,
        'improvement': {
            'speedup': speedup,
            'performance_gain_pct': (speedup - 1) * 100,
            'throughput_increase': throughput_gain,
            'throughput_gain_pct': throughput_gain_pct
        }
    }

    with open('opt125m_pinned_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to opt125m_pinned_results.json")

    return results

if __name__ == "__main__":
    results = main()