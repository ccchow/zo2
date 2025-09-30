# Copyright (c) 2025 ZO2 Contributors
# Licensed under the Apache License, Version 2.0

"""
Multi-GPU utilities for pipeline parallelism in ZO2.

This module provides utilities for:
- Detecting GPU topology (NVLink connectivity)
- Computing optimal layer distribution across pipeline stages
- Managing stage-to-device mappings
- Validating GPU availability and memory
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import subprocess
import re
import time


def detect_nvlink_topology() -> Dict[int, List[int]]:
    """
    Detect NVLink connectivity between GPUs.

    Returns:
        Dictionary mapping GPU ID to list of directly connected GPU IDs via NVLink.

    Example:
        {0: [1], 1: [0], 2: [3], 3: [2]}  # GPU 0-1 paired, GPU 2-3 paired

    Note:
        On RTX 3090, typically only 2-way NVLink bridges are available.
        Pairs communicate at ~56 GB/s, while cross-pair is PCIe 4.0 ~32 GB/s.
    """
    nvlink_map = {}

    if not torch.cuda.is_available():
        return nvlink_map

    num_gpus = torch.cuda.device_count()

    try:
        # Try nvidia-smi to detect NVLink
        result = subprocess.run(
            ['nvidia-smi', 'nvlink', '--status'],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Parse nvidia-smi nvlink output
        # Format: "GPU 0: ... Link N: <GPU_ID>"
        for line in result.stdout.split('\n'):
            match = re.search(r'GPU (\d+):.*Link \d+:.*(\d+)', line)
            if match:
                gpu_id = int(match.group(1))
                connected_gpu = int(match.group(2))
                if gpu_id not in nvlink_map:
                    nvlink_map[gpu_id] = []
                if connected_gpu not in nvlink_map[gpu_id]:
                    nvlink_map[gpu_id].append(connected_gpu)

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Warning: Could not detect NVLink topology via nvidia-smi: {e}")
        print("Falling back to p2p accessibility check")

        # Fallback: use PyTorch P2P check
        for i in range(num_gpus):
            nvlink_map[i] = []
            for j in range(num_gpus):
                if i != j and torch.cuda.can_device_access_peer(i, j):
                    nvlink_map[i].append(j)

    return nvlink_map


def get_optimal_stage_order(num_stages: int, nvlink_map: Dict[int, List[int]]) -> List[int]:
    """
    Compute optimal GPU ordering for pipeline stages to maximize NVLink usage.

    Args:
        num_stages: Number of pipeline stages
        nvlink_map: NVLink connectivity mapping from detect_nvlink_topology()

    Returns:
        List of GPU IDs ordered to maximize NVLink adjacency

    Example:
        For 4 GPUs with NVLink pairs (0-1, 2-3), returns [0, 1, 2, 3]
        This ensures stages 0→1 and 2→3 communicate over NVLink.
    """
    if not nvlink_map or num_stages == 1:
        return list(range(num_stages))

    # Find NVLink pairs
    pairs = []
    visited = set()

    for gpu_id, connected in nvlink_map.items():
        if gpu_id in visited:
            continue
        if connected:
            # Found a pair
            peer = connected[0]
            pairs.append((gpu_id, peer))
            visited.add(gpu_id)
            visited.add(peer)

    # Build stage order: try to keep adjacent stages on same NVLink pair
    stage_order = []
    for pair in pairs:
        stage_order.extend(pair)

    # Add any remaining GPUs
    all_gpus = set(range(num_stages))
    remaining = all_gpus - visited
    stage_order.extend(sorted(remaining))

    return stage_order[:num_stages]


def profile_layer_compute_time(
    model: nn.Module,
    layer_ids: List[int],
    device: str,
    seq_len: int = 128,
    batch_size: int = 1,
    vocab_size: int = 50272,
    hidden_size: int = 768,
    warmup_steps: int = 3,
    measure_steps: int = 10
) -> List[float]:
    """
    Profile per-layer computation time for auto-balanced distribution.

    Args:
        model: The model with layers to profile
        layer_ids: List of layer indices to profile
        device: Device to run profiling on
        seq_len: Sequence length for profiling
        batch_size: Batch size for profiling
        vocab_size: Vocabulary size for dummy inputs
        hidden_size: Hidden state size
        warmup_steps: Number of warmup iterations
        measure_steps: Number of measurement iterations

    Returns:
        List of average times (seconds) per layer
    """
    layer_times = []

    # Create dummy input
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size,
        device=device,
        dtype=torch.float32
    )

    for layer_id in layer_ids:
        try:
            layer = model.layers[layer_id].to(device)
            layer.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(warmup_steps):
                    _ = layer(hidden_states)
                    torch.cuda.synchronize()

            # Measure
            times = []
            with torch.no_grad():
                for _ in range(measure_steps):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = layer(hidden_states)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            layer_times.append(avg_time)

            # Clean up
            layer.cpu()
            del layer
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warning: Could not profile layer {layer_id}: {e}")
            layer_times.append(0.001)  # Fallback estimate

    return layer_times


def calculate_layer_distribution(
    num_layers: int,
    num_stages: int,
    strategy: str = 'balanced',
    layer_times: Optional[List[float]] = None
) -> List[List[int]]:
    """
    Calculate how to distribute layers across pipeline stages.

    Args:
        num_layers: Total number of transformer layers
        num_stages: Number of pipeline stages (GPUs)
        strategy: 'balanced' (equal counts) or 'auto' (compute-balanced)
        layer_times: Per-layer compute times (required for 'auto')

    Returns:
        List of layer ID lists per stage

    Example:
        For 40 layers, 4 stages, 'balanced':
        [[0..9], [10..19], [20..29], [30..39]]
    """
    if strategy == 'balanced':
        # Equal layer counts
        layers_per_stage = num_layers // num_stages
        remainder = num_layers % num_stages

        stage_layers = []
        start = 0
        for i in range(num_stages):
            # Distribute remainder across first few stages
            count = layers_per_stage + (1 if i < remainder else 0)
            stage_layers.append(list(range(start, start + count)))
            start += count

        return stage_layers

    elif strategy == 'auto':
        if layer_times is None or len(layer_times) != num_layers:
            print("Warning: layer_times not provided or incorrect length, falling back to 'balanced'")
            return calculate_layer_distribution(num_layers, num_stages, 'balanced')

        # Greedy partition to balance compute time
        # Sort layers by time (descending) and assign to least-loaded stage
        layer_indices = list(range(num_layers))
        stage_times = [0.0] * num_stages
        stage_layers = [[] for _ in range(num_stages)]

        # Greedy: assign each layer to stage with minimum current time
        for layer_id in layer_indices:
            min_stage = min(range(num_stages), key=lambda s: stage_times[s])
            stage_layers[min_stage].append(layer_id)
            stage_times[min_stage] += layer_times[layer_id]

        # Sort layer IDs within each stage
        for stage in stage_layers:
            stage.sort()

        return stage_layers

    else:
        raise ValueError(f"Unknown distribution strategy: {strategy}")


def get_layer_to_device_mapping(
    stage_layers: List[List[int]],
    stage_devices: List[str],
    tie_word_embeddings: bool,
    embedding_stage: int = 0
) -> Tuple[Dict[int, str], int, int]:
    """
    Create mapping from layer IDs to device, ensuring embed/lm_head co-location if tied.

    Args:
        stage_layers: Layer IDs per stage from calculate_layer_distribution()
        stage_devices: Device strings per stage, e.g., ['cuda:0', 'cuda:1']
        tie_word_embeddings: If True, embed_tokens and lm_head must be on same device
        embedding_stage: Which stage should host embeddings (0=first, -1=last)

    Returns:
        (layer_to_device, embed_device_stage, lm_head_device_stage)

    Note:
        If tie_word_embeddings=True, embed and lm_head are both placed on embedding_stage device.
        If False, embed is on embedding_stage, lm_head is on last stage.
    """
    layer_to_device = {}

    for stage_idx, layers in enumerate(stage_layers):
        device = stage_devices[stage_idx]
        for layer_id in layers:
            layer_to_device[layer_id] = device

    # Determine embed/lm_head placement
    if embedding_stage == -1:
        embedding_stage = len(stage_devices) - 1

    embed_device_stage = embedding_stage

    if tie_word_embeddings:
        # Both on same stage
        lm_head_device_stage = embedding_stage
    else:
        # LM head on last stage (typical for pipeline)
        lm_head_device_stage = len(stage_devices) - 1

    return layer_to_device, embed_device_stage, lm_head_device_stage


def validate_gpu_availability(num_gpus: int, gpu_devices: Optional[List[str]] = None) -> List[str]:
    """
    Validate that requested GPUs are available and return device list.

    Args:
        num_gpus: Number of GPUs requested
        gpu_devices: Optional explicit device list

    Returns:
        List of validated device strings

    Raises:
        RuntimeError: If GPUs are not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot use multi-GPU pipeline")

    available_gpus = torch.cuda.device_count()

    if num_gpus > available_gpus:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only {available_gpus} are available"
        )

    if gpu_devices is not None:
        # Validate explicit device list
        if len(gpu_devices) != num_gpus:
            raise ValueError(
                f"gpu_devices list length ({len(gpu_devices)}) doesn't match num_gpus ({num_gpus})"
            )

        # Validate device strings
        for device_str in gpu_devices:
            try:
                device = torch.device(device_str)
                if device.type != 'cuda':
                    raise ValueError(f"Device {device_str} is not a CUDA device")
                if device.index >= available_gpus:
                    raise ValueError(f"Device {device_str} index exceeds available GPUs")
            except Exception as e:
                raise ValueError(f"Invalid device string {device_str}: {e}")

        return gpu_devices
    else:
        # Auto-generate device list with optimal ordering
        nvlink_map = detect_nvlink_topology()
        stage_order = get_optimal_stage_order(num_gpus, nvlink_map)
        return [f'cuda:{gpu_id}' for gpu_id in stage_order]


def estimate_memory_per_gpu(
    model: nn.Module,
    stage_layers: List[List[int]],
    batch_size: int = 1,
    seq_len: int = 512,
    hidden_size: int = 768,
    dtype: torch.dtype = torch.float16
) -> List[float]:
    """
    Estimate memory usage per GPU for pipeline stages.

    Args:
        model: The model to estimate
        stage_layers: Layer distribution per stage
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden state dimension
        dtype: Data type for activations

    Returns:
        List of estimated memory in MB per stage

    Note:
        Assumes ZO2-style inference memory footprint (forward-only, streaming layers).
    """
    bytes_per_element = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    estimates = []

    for stage_idx, layers in enumerate(stage_layers):
        # Parameter memory (layers only, streaming assumption)
        param_memory = 0
        for layer_id in layers:
            try:
                layer = model.layers[layer_id]
                param_memory += sum(p.numel() * bytes_per_element for p in layer.parameters())
            except:
                # Fallback estimate: ~100M params per layer for large models
                param_memory += 100e6 * bytes_per_element

        # Activation memory (hidden states + attention)
        activation_memory = (
            batch_size * seq_len * hidden_size * bytes_per_element * 4  # hidden, attn_out, ffn intermediate
        )

        # Micro-batch buffer (assume 2x for double buffering)
        buffer_memory = activation_memory * 2

        total_mb = (param_memory + activation_memory + buffer_memory) / (1024 ** 2)
        estimates.append(total_mb)

    return estimates


def print_pipeline_summary(
    stage_layers: List[List[int]],
    stage_devices: List[str],
    embed_stage: int,
    lm_head_stage: int,
    nvlink_map: Optional[Dict[int, List[int]]] = None
):
    """
    Print a summary of the pipeline configuration.

    Args:
        stage_layers: Layer distribution per stage
        stage_devices: Device list
        embed_stage: Stage hosting embeddings
        lm_head_stage: Stage hosting LM head
        nvlink_map: NVLink topology (optional)
    """
    print("\n" + "="*70)
    print("PIPELINE PARALLELISM CONFIGURATION")
    print("="*70)

    for stage_idx, (layers, device) in enumerate(zip(stage_layers, stage_devices)):
        components = []
        if stage_idx == embed_stage:
            components.append("embed_tokens")
        if stage_idx == lm_head_stage:
            components.append("lm_head")

        components_str = f" + {', '.join(components)}" if components else ""
        layer_range = f"[{min(layers)}..{max(layers)}]" if layers else "[]"

        print(f"Stage {stage_idx} ({device}): Layers {layer_range} ({len(layers)} layers){components_str}")

    if nvlink_map:
        print("\nNVLink Topology:")
        for gpu_id, connected in nvlink_map.items():
            if connected:
                print(f"  GPU {gpu_id} ↔ {connected} (NVLink)")
            else:
                print(f"  GPU {gpu_id}: PCIe only")

    print("="*70 + "\n")
