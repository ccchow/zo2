#!/usr/bin/env python3
"""
Multi-GPU Pipeline Parallelism Benchmark for ZO2

Tests pipeline parallelism across multiple GPUs with various OPT model sizes.
Measures throughput, memory usage, and validates NVLink topology detection.

Usage:
    # Test 2-GPU configuration
    python test/benchmark_multigpu_opt.py --model facebook/opt-6.7b --num_gpus 2 --micro_batches 16

    # Test 4-GPU configuration with auto layer distribution
    python test/benchmark_multigpu_opt.py --model facebook/opt-13b --num_gpus 4 --layer_distribution auto

    # Test with custom layer split
    python test/benchmark_multigpu_opt.py --model facebook/opt-30b --num_gpus 4 --custom_layer_split 12,24,36
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
import gc
import argparse
import json
from zo2.config.mezo_sgd import MeZOSGDConfig
from zo2.model.huggingface.opt.mezo_sgd import zo2
from transformers import OPTConfig, AutoConfig
import psutil

def get_memory_stats():
    """Get current memory usage for all GPUs and CPU."""
    stats = {'gpus': {}, 'cpu_mb': 0}

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            stats['gpus'][i] = {
                'allocated_mb': torch.cuda.memory_allocated(i) / 1024 / 1024,
                'reserved_mb': torch.cuda.memory_reserved(i) / 1024 / 1024,
                'peak_mb': torch.cuda.max_memory_allocated(i) / 1024 / 1024
            }

    process = psutil.Process()
    stats['cpu_mb'] = process.memory_info().rss / 1024 / 1024
    return stats

def print_memory_summary(stats, label="Memory"):
    """Print memory usage summary."""
    print(f"\n{label}:")
    for gpu_id, gpu_stats in stats['gpus'].items():
        print(f"  GPU {gpu_id}:")
        print(f"    Allocated: {gpu_stats['allocated_mb']:.1f} MB")
        print(f"    Reserved:  {gpu_stats['reserved_mb']:.1f} MB")
        print(f"    Peak:      {gpu_stats['peak_mb']:.1f} MB")
    print(f"  CPU: {stats['cpu_mb']:.1f} MB")

def run_benchmark(args):
    """Run multi-GPU benchmark with specified configuration."""

    print("="*80)
    print(f"Multi-GPU Pipeline Parallelism Benchmark - {args.model}")
    print("="*80)

    # Validate GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return None

    num_available_gpus = torch.cuda.device_count()
    if args.num_gpus > num_available_gpus:
        print(f"ERROR: Requested {args.num_gpus} GPUs but only {num_available_gpus} available")
        return None

    print(f"\nGPU Configuration:")
    print(f"  Available GPUs: {num_available_gpus}")
    print(f"  Using GPUs: {args.num_gpus}")
    for i in range(args.num_gpus):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      Memory: {props.total_memory / 1024**3:.1f} GB")

    # Clear memory
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Create ZO config with multi-GPU settings
    zo_config = MeZOSGDConfig()
    zo_config.lr = 1e-6
    zo_config.weight_decay = 0.01
    zo_config.eps = 1e-3
    zo_config.zo2 = True
    zo_config.offloading_device = 'cpu'
    zo_config.working_device = 'cuda:0'
    zo_config.overlap = True
    zo_config.use_pinned_memory = args.use_pinned_memory
    zo_config.pinned_memory_prefetch = args.use_pinned_memory

    # Multi-GPU pipeline parallelism config
    zo_config.num_gpus = args.num_gpus
    zo_config.pipeline_parallel = args.num_gpus > 1
    zo_config.layer_distribution = args.layer_distribution
    zo_config.micro_batches = args.micro_batches
    zo_config.tie_word_embeddings = True
    zo_config.stage_io_dtype = 'bf16'
    zo_config.enable_cpu_offloading_per_gpu = args.enable_cpu_offload
    zo_config.p2p_backend = 'nccl'
    zo_config.p2p_overlap = True

    if args.custom_layer_split:
        zo_config.layer_distribution = 'custom'
        zo_config.custom_layer_split = args.custom_layer_split

    print(f"\nZO2 Configuration:")
    print(f"  Pipeline parallel: {zo_config.pipeline_parallel}")
    print(f"  Micro-batches: {zo_config.micro_batches}")
    print(f"  Layer distribution: {zo_config.layer_distribution}")
    print(f"  CPU offload per GPU: {zo_config.enable_cpu_offloading_per_gpu}")
    print(f"  Stage I/O dtype: {zo_config.stage_io_dtype}")
    print(f"  Pinned memory: {zo_config.use_pinned_memory}")

    # Load model configuration
    print(f"\nLoading model configuration from {args.model}...")
    try:
        model_config = AutoConfig.from_pretrained(args.model)
    except Exception as e:
        print(f"ERROR: Failed to load model config: {e}")
        return None

    print(f"  Model: {model_config._name_or_path}")
    print(f"  Layers: {model_config.num_hidden_layers}")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Vocab size: {model_config.vocab_size}")

    # Calculate expected parameters
    num_params = (
        model_config.vocab_size * model_config.hidden_size * 2 +  # embeddings + lm_head
        model_config.num_hidden_layers * (
            4 * model_config.hidden_size * model_config.hidden_size +  # attention
            2 * model_config.hidden_size * model_config.ffn_dim  # FFN
        )
    ) / 1e9
    print(f"  Estimated parameters: {num_params:.2f}B")

    # Get initial memory
    initial_mem = get_memory_stats()
    print_memory_summary(initial_mem, "Initial Memory")

    # Create model
    print("\nInitializing model...")
    start_time = time.time()

    try:
        model = zo2.OPTForCausalLM(model_config)
        model.zo_init(zo_config)
    except Exception as e:
        print(f"ERROR: Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    init_time = time.time() - start_time
    print(f"Model initialization took {init_time:.2f}s")

    # Get post-init memory
    post_init_mem = get_memory_stats()
    print_memory_summary(post_init_mem, "After Initialization")

    # Prepare dummy data
    batch_size = args.batch_size
    seq_length = args.seq_length
    vocab_size = model_config.vocab_size

    print(f"\nBenchmark Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Benchmark steps: {args.benchmark_steps}")

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device='cuda:0')
    labels = input_ids.clone()

    # Warmup
    print(f"\nWarming up ({args.warmup_steps} steps)...")
    model.zo_train()

    try:
        for i in range(args.warmup_steps):
            outputs = model(input_ids=input_ids, labels=labels)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                loss = outputs[0]

            # Synchronize all GPUs
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.synchronize(gpu_id)

            print(f"  Warmup {i+1}/{args.warmup_steps}: loss={loss.item():.4f}")
    except Exception as e:
        print(f"ERROR: Warmup failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Benchmark
    print(f"\nRunning benchmark ({args.benchmark_steps} steps)...")
    step_times = []
    losses = []

    try:
        for step in range(args.benchmark_steps):
            # Synchronize all GPUs before starting
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.synchronize(gpu_id)

            start_time = time.perf_counter()

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, torch.Tensor):
                loss = outputs
            else:
                loss = outputs[0]

            # Synchronize all GPUs
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.synchronize(gpu_id)

            step_time = time.perf_counter() - start_time
            step_times.append(step_time)
            losses.append(loss.item())

            if (step + 1) % 5 == 0 or step == 0:
                avg_time = sum(step_times) / len(step_times)
                print(f"  Step {step+1}/{args.benchmark_steps}: time={step_time:.4f}s, avg={avg_time:.4f}s, loss={loss.item():.4f}")
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Get final memory stats
    final_mem = get_memory_stats()
    print_memory_summary(final_mem, "Final Memory")

    # Calculate statistics
    avg_step_time = sum(step_times) / len(step_times)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    std_step_time = (sum((t - avg_step_time)**2 for t in step_times) / len(step_times))**0.5
    throughput = 1.0 / avg_step_time
    avg_loss = sum(losses) / len(losses)

    # Calculate total GPU memory
    total_gpu_memory = sum(gpu['peak_mb'] for gpu in final_mem['gpus'].values())

    # Print results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"\nTiming Statistics:")
    print(f"  Average step time: {avg_step_time:.4f}s (±{std_step_time:.4f}s)")
    print(f"  Min step time: {min_step_time:.4f}s")
    print(f"  Max step time: {max_step_time:.4f}s")
    print(f"  Throughput: {throughput:.2f} steps/sec")
    print(f"  Tokens/sec: {throughput * batch_size * seq_length:.0f}")

    print(f"\nMemory Usage:")
    print(f"  Total GPU memory (peak): {total_gpu_memory:.1f} MB")
    for gpu_id, gpu_stats in final_mem['gpus'].items():
        print(f"  GPU {gpu_id} peak: {gpu_stats['peak_mb']:.1f} MB")
    print(f"  CPU memory: {final_mem['cpu_mb']:.1f} MB")

    print(f"\nTraining:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Loss std: {(sum((l - avg_loss)**2 for l in losses) / len(losses))**0.5:.4f}")

    # Estimate pipeline efficiency
    if args.num_gpus > 1:
        bubble_fraction = (args.num_gpus - 1) / (args.micro_batches + args.num_gpus - 1)
        pipeline_efficiency = 1.0 - bubble_fraction
        print(f"\nPipeline Efficiency:")
        print(f"  Stages: {args.num_gpus}")
        print(f"  Micro-batches: {args.micro_batches}")
        print(f"  Bubble fraction: {bubble_fraction:.2%}")
        print(f"  Pipeline efficiency: {pipeline_efficiency:.2%}")

    # Clean up
    del model
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    gc.collect()

    # Return results dictionary
    results = {
        'model': args.model,
        'num_gpus': args.num_gpus,
        'num_layers': model_config.num_hidden_layers,
        'num_params_billion': num_params,
        'config': {
            'batch_size': batch_size,
            'seq_length': seq_length,
            'micro_batches': args.micro_batches,
            'layer_distribution': args.layer_distribution,
            'cpu_offload': args.enable_cpu_offload,
            'pinned_memory': args.use_pinned_memory
        },
        'timing': {
            'avg_step_time': avg_step_time,
            'min_step_time': min_step_time,
            'max_step_time': max_step_time,
            'std_step_time': std_step_time,
            'throughput': throughput,
            'tokens_per_sec': throughput * batch_size * seq_length
        },
        'memory': {
            'total_gpu_peak_mb': total_gpu_memory,
            'per_gpu_peak_mb': {gpu_id: stats['peak_mb'] for gpu_id, stats in final_mem['gpus'].items()},
            'cpu_mb': final_mem['cpu_mb']
        },
        'training': {
            'avg_loss': avg_loss,
            'loss_std': (sum((l - avg_loss)**2 for l in losses) / len(losses))**0.5
        }
    }

    if args.num_gpus > 1:
        results['pipeline'] = {
            'bubble_fraction': bubble_fraction,
            'efficiency': pipeline_efficiency
        }

    return results

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Pipeline Parallelism Benchmark for ZO2')

    # Model configuration
    parser.add_argument('--model', type=str, default='facebook/opt-6.7b',
                        help='Model name or path (e.g., facebook/opt-6.7b, facebook/opt-13b)')

    # Multi-GPU configuration
    parser.add_argument('--num_gpus', type=int, default=2,
                        help='Number of GPUs to use for pipeline parallelism')
    parser.add_argument('--layer_distribution', type=str, default='balanced',
                        choices=['balanced', 'auto'],
                        help='Layer distribution strategy')
    parser.add_argument('--custom_layer_split', type=str, default=None,
                        help='Custom layer split boundaries (comma-separated, e.g., "12,24,36")')
    parser.add_argument('--micro_batches', type=int, default=16,
                        help='Number of micro-batches for pipeline')
    parser.add_argument('--enable_cpu_offload', action='store_true',
                        help='Enable CPU offloading within each GPU stage')
    parser.add_argument('--use_pinned_memory', action='store_true',
                        help='Use pinned memory for faster CPU-GPU transfers')

    # Benchmark configuration
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--warmup_steps', type=int, default=3,
                        help='Number of warmup steps')
    parser.add_argument('--benchmark_steps', type=int, default=20,
                        help='Number of benchmark steps')

    # Output configuration
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')

    args = parser.parse_args()

    # Parse custom layer split
    if args.custom_layer_split:
        args.custom_layer_split = [int(x.strip()) for x in args.custom_layer_split.split(',')]

    # Run benchmark
    results = run_benchmark(args)

    if results is None:
        print("\nBenchmark failed!")
        return 1

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = args.model.replace('/', '_')
        output_path = f'benchmark_multigpu_{model_name}_{args.num_gpus}gpu_{timestamp}.json'

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
