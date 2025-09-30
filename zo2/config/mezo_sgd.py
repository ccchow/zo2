# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MeZOSGDConfig:
    # zo method
    zo_method: str = "mezo-sgd" # zo method name, every zo config must include this attribute

    # zo config
    lr: float = 1e-3
    weight_decay: float = 1e-1
    eps: float = 1e-3
    max_zo_random_seed = 1000000000

    # zo2 config
    zo2: bool = True    # use offloading or not
    offloading_blocks: list = None  # specify offloading blocks or not
    offloading_device: str = 'cpu'  # offload device, can be CPU or a path (for disk offloading, but currently unavailable)
    working_device: str = 'cuda'    # compute device, can be any CUDA device
    overlap: bool = True    # use scheduler to overlap or not
    use_pinned_memory: bool = False    # use pinned memory for faster CPU-GPU transfers (20-30% speedup)
    pinned_memory_prefetch: bool = True    # prefetch next module to pinned memory while computing current
    compute_module_optimize_method: str = ''   # possible values are: ['', 'torch.compile']
    compute_function_optimize_method: str = ''   # possible values are: ['', 'torch.jit.script']
    communicate_optimize_method: str = ''   # possible values are: ['', 'bucket']
    amp: bool = False   # use amp or not
    amp_precision: torch.dtype = torch.bfloat16 # amp autocast precision, possible values are: [torch.bfloat16, torch.float32], valid when using amp
    precision_on_offloading_device: torch.dtype = torch.float16 # precision on offloading device, valid when using amp
    precision_on_working_device: torch.dtype = torch.float32    # precision on working device, valid when using amp
    amp_compress_method: str = 'naive'  # currently only support naive amp compress, valid when using amp

    # multi-GPU pipeline parallelism config
    num_gpus: int = 1    # number of GPUs to use (1 = single GPU mode)
    pipeline_parallel: bool = False    # enable pipeline parallelism across GPUs
    gpu_devices: Optional[List[str]] = None    # explicit GPU device list, e.g., ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    layer_distribution: str = 'auto'    # strategy: 'balanced' (equal layer counts), 'auto' (compute-balanced), 'custom'
    custom_layer_split: Optional[List[int]] = None    # custom layer boundaries per GPU, e.g., [10, 20, 30] for 4 GPUs
    micro_batches: int = 16    # number of micro-batches for pipeline (>=12 for 4 stages to keep bubble <20%)
    pipeline_schedule: str = 'forward_fill_drain'    # GPipe-style forward-only scheduling for ZO2
    tie_word_embeddings: bool = True    # if True, enforce embed_tokens and lm_head co-location
    stage_io_dtype: str = 'bf16'    # dtype for inter-stage hidden state transfers ('bf16' or 'fp16'), Ampere supports bf16
    enable_cpu_offloading_per_gpu: bool = True    # still use CPU offload within each GPU pipeline stage
    p2p_backend: str = 'nccl'    # backend for stage-to-stage communication ('nccl' recommended)
    p2p_overlap: bool = True    # use separate transfer stream + events for overlapped P2P transfers

    # debug
    debug_mode: bool = False    # set 'True' to disable random noise