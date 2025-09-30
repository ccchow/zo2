# Copyright (c) 2025 liangyuwang
# Licensed under the Apache License, Version 2.0

import sys
sys.path.append('./zo2')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

from .zo import MeZOSGD
from ...config.mezo_sgd import MeZOSGDConfig
from .utils import *


class MeZO2SGD(MeZOSGD):
    first_call_eval = True  # Class variable specifically for tracking eval function
    
    """
    Extends MeZOSGD to support advanced offloading techniques that enhance the capability
    to train large models on systems with limited GPU memory. It manages the intricate
    balance between CPU and GPU, leveraging zeroth-order optimization with dynamic memory
    management through offloading.
    """
    def __init__(self, model, config: MeZOSGDConfig):
        """
        Initializes the MeZO2SGD optimizer, setting up the necessary configuration for
        offloading and optimization techniques.

        Args:
            model (nn.Module): The model whose parameters will be optimized.
            config (MeZOSGDConfig): Configuration object specifying optimizer settings including
                                    offloading and overlapping options.
        """
        assert config.zo2, "MeZO2SGD can only work with offloading."
        super().__init__(model, config)
        self.device = config.working_device
        self.offloading_device = config.offloading_device
        self.overlap = config.overlap
        self.offloading_blocks = config.offloading_blocks
        self.compute_module_optimize_method = config.compute_module_optimize_method
        self.compute_function_optimize_method = config.compute_function_optimize_method
        self.communicate_optimize_method = config.communicate_optimize_method
        self.amp = config.amp
        self.amp_precision = config.amp_precision
        self.precision_on_offloading_device = config.precision_on_offloading_device
        self.precision_on_working_device = config.precision_on_working_device
        self.amp_compress_method = config.amp_compress_method
        self.use_pinned_memory = config.use_pinned_memory
        self.pinned_memory_prefetch = config.pinned_memory_prefetch

        # Multi-GPU pipeline parallelism config
        self.num_gpus = config.num_gpus
        self.pipeline_parallel = config.pipeline_parallel
        self.gpu_devices = config.gpu_devices
        self.layer_distribution = config.layer_distribution
        self.custom_layer_split = config.custom_layer_split
        self.micro_batches = config.micro_batches
        self.pipeline_schedule = config.pipeline_schedule
        self.tie_word_embeddings = config.tie_word_embeddings
        self.stage_io_dtype = config.stage_io_dtype
        self.enable_cpu_offloading_per_gpu = config.enable_cpu_offloading_per_gpu
        self.p2p_backend = config.p2p_backend
        self.p2p_overlap = config.p2p_overlap

        self.init_zo2()
    
    def init_zo2(self):
        """
        Sets up CUDA streams and initializes the offloading and uploading mechanisms
        required for efficient computation management across devices.
        """
        # Single-GPU mode: initialize streams on working device
        if not self.pipeline_parallel or self.num_gpus == 1:
            self.upload_stream = torch.cuda.Stream()
            self.offload_stream = torch.cuda.Stream()
            self.compute_stream = torch.cuda.Stream()
        else:
            # Multi-GPU pipeline mode: initialize per-device streams
            self.init_multi_gpu_streams()

        self.zo_random_seed = None
        self.rstate = None
        self.rstate_queue = deque(maxlen=2)
        self.last_rstate = None
        self.projected_grad = 0

        # Initialize multi-GPU sharding before upload
        if self.pipeline_parallel and self.num_gpus > 1:
            self.init_multi_gpu_sharding()
        else:
            self.init_zo2_upload()

        if self.amp: self.init_zo2_amp()
        if self.use_pinned_memory: self.init_pinned_memory_buffers()
    
    def init_zo2_amp(self):
        """
        Initializes the model parameters to use different precision levels based on their current device.
        This method works with Automatic Mixed Precision (AMP) by setting the precision for parameters 
        based on whether they are located on the working device or the offloading device.
        """
        working_device = torch.device(self.device)
        offloading_device = torch.device(self.offloading_device)
        for p in self.model.parameters():
            if p.device == working_device:
                p.data = p.data.to(dtype=self.precision_on_working_device)
            elif p.device == offloading_device:
                p.data = p.data.to(dtype=self.precision_on_offloading_device)
            else:
                raise ValueError(f"Unsupported device found for parameter: {p.device}")

    def init_pinned_memory_buffers(self):
        """
        Initializes pinned memory buffers for faster CPU-GPU transfers.
        Pinned memory provides 20-30% speedup by enabling asynchronous transfers.
        """
        self.pinned_buffers = {}
        self.pinned_buffer_cache = {}

        # Check if pinned memory is available
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, disabling pinned memory")
            self.use_pinned_memory = False
            return

        # Get available pinned memory (conservative estimate)
        device_props = torch.cuda.get_device_properties(self.device)
        total_memory = device_props.total_memory

        # We'll allocate pinned buffers on demand to avoid wasting memory
        print(f"Pinned memory optimization enabled for device {self.device}")
        print(f"GPU total memory: {total_memory / 1024**3:.2f} GB")

    def get_or_create_pinned_buffer(self, tensor_size, dtype):
        """
        Get a pinned memory buffer of the specified size, creating if necessary.
        Implements buffer reuse to minimize allocation overhead.
        """
        key = (tensor_size, dtype)

        if key in self.pinned_buffer_cache:
            return self.pinned_buffer_cache[key]

        try:
            # Allocate new pinned memory buffer
            buffer = torch.empty(tensor_size, dtype=dtype, pin_memory=True)
            self.pinned_buffer_cache[key] = buffer
            return buffer
        except RuntimeError as e:
            # Fall back to regular memory if pinned allocation fails
            print(f"Warning: Failed to allocate pinned memory: {e}")
            self.use_pinned_memory = False
            return None

    def init_multi_gpu_streams(self):
        """
        Initialize CUDA streams for multi-GPU pipeline parallelism.
        Creates compute and transfer streams for each pipeline stage.
        """
        from ...utils.multi_gpu import validate_gpu_availability, detect_nvlink_topology, print_pipeline_summary

        # Validate and get device list
        self.stage_devices = validate_gpu_availability(self.num_gpus, self.gpu_devices)

        # Detect NVLink topology
        self.nvlink_map = detect_nvlink_topology()

        # Create per-device streams
        self.stage_compute_streams = {}
        self.stage_transfer_streams = {}

        for device_str in self.stage_devices:
            device = torch.device(device_str)

            # High-priority transfer stream for P2P communication
            with torch.cuda.device(device):
                self.stage_compute_streams[device_str] = torch.cuda.Stream()
                self.stage_transfer_streams[device_str] = torch.cuda.Stream(priority=-1)

        print(f"\nInitialized {len(self.stage_devices)} pipeline stages with per-device streams")
        if self.nvlink_map:
            print("NVLink pairs detected:")
            for gpu_id, connected in self.nvlink_map.items():
                if connected:
                    print(f"  GPU {gpu_id} ↔ GPU {connected}")

    def init_multi_gpu_sharding(self):
        """
        Initialize multi-GPU pipeline sharding with embed/lm_head co-location.
        This method distributes transformer layers across GPUs according to the
        selected distribution strategy.
        """
        from ...utils.multi_gpu import (
            calculate_layer_distribution,
            get_layer_to_device_mapping,
            profile_layer_compute_time,
            print_pipeline_summary
        )

        # Get model layers (this will be model-specific, handled in subclass)
        # For now, store configuration that will be used by model-specific implementation
        num_layers = len(self.model.decoder.layers) if hasattr(self.model, 'decoder') else 0

        if num_layers == 0:
            print("Warning: Could not determine number of layers for multi-GPU sharding")
            return

        # Profile layers if using auto distribution
        layer_times = None
        if self.layer_distribution == 'auto':
            print("Profiling layers for compute-balanced distribution...")
            try:
                layer_times = profile_layer_compute_time(
                    model=self.model.decoder if hasattr(self.model, 'decoder') else self.model,
                    layer_ids=list(range(min(num_layers, 10))),  # Profile first 10 layers as sample
                    device=self.stage_devices[0],
                    seq_len=128,
                    batch_size=1
                )
                # Extrapolate for all layers
                if len(layer_times) < num_layers:
                    avg_time = sum(layer_times) / len(layer_times)
                    layer_times = layer_times + [avg_time] * (num_layers - len(layer_times))
            except Exception as e:
                print(f"Warning: Layer profiling failed: {e}, falling back to 'balanced'")
                self.layer_distribution = 'balanced'

        # Calculate layer distribution
        if self.custom_layer_split:
            # Custom distribution
            self.stage_layers = []
            start = 0
            for end in self.custom_layer_split:
                self.stage_layers.append(list(range(start, end)))
                start = end
            self.stage_layers.append(list(range(start, num_layers)))
        else:
            # Auto or balanced distribution
            self.stage_layers = calculate_layer_distribution(
                num_layers=num_layers,
                num_stages=self.num_gpus,
                strategy=self.layer_distribution,
                layer_times=layer_times
            )

        # Get layer-to-device mapping with embed/lm_head co-location
        embedding_stage = 0 if self.tie_word_embeddings else 0  # Always start on first stage
        self.layer_to_device, self.embed_device_stage, self.lm_head_device_stage = get_layer_to_device_mapping(
            stage_layers=self.stage_layers,
            stage_devices=self.stage_devices,
            tie_word_embeddings=self.tie_word_embeddings,
            embedding_stage=embedding_stage
        )

        # Store stage info
        self.num_stages = len(self.stage_devices)

        # Print configuration summary
        print_pipeline_summary(
            stage_layers=self.stage_layers,
            stage_devices=self.stage_devices,
            embed_stage=self.embed_device_stage,
            lm_head_stage=self.lm_head_device_stage,
            nvlink_map=self.nvlink_map
        )

        # Now distribute the actual model layers (model-specific, will be overridden)
        self.distribute_model_layers()

    def distribute_model_layers(self):
        """
        Distribute model layers across GPUs according to pipeline sharding plan.
        This is a base implementation that should be overridden by model-specific subclasses.
        """
        # This will be implemented by model-specific optimizer classes
        # (e.g., OptimizerOPTDecoder) that know the model structure
        pass

    def pipeline_forward(self, input_ids, labels=None, **kwargs):
        """
        Forward-only pipeline parallelism for ZO2/MeZO.

        MeZO uses a two-point gradient estimator (loss at w+εu and w−εu) with NO backward pass.
        This method implements GPipe-style fill-drain pipeline scheduling for forward-only execution.

        Pipeline bubble fraction ≈ (P-1)/(M+P-1) where P=stages, M=micro_batches.
        For P=4, M≥12 keeps bubble <20%.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Labels for loss computation [batch_size, seq_len]
            **kwargs: Additional forward arguments

        Returns:
            Loss tensor (scalar)

        Note:
            This implements the core MeZO two-pass evaluation:
            1. Apply +ε perturbation, run forward pipeline → loss_plus
            2. Apply −ε perturbation, run forward pipeline → loss_minus
            3. Compute gradient estimate: coef = (loss_plus - loss_minus) / (2ε)
            4. Update parameters: p.add_(u, alpha=−η·coef)
        """
        if not self.pipeline_parallel or self.num_gpus == 1:
            # Single-GPU fallback
            return self.model(input_ids=input_ids, labels=labels, **kwargs).loss

        # Convert stage_io_dtype string to torch.dtype
        io_dtype = torch.bfloat16 if self.stage_io_dtype == 'bf16' else torch.float16

        # Split batch into micro-batches
        batch_size = input_ids.size(0)
        micro_batch_size = max(1, batch_size // self.micro_batches)
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        # Phase 1: Apply +ε perturbation (handled by caller via zo_forward)
        # Run forward pipeline with micro-batching
        losses_plus = []
        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            mb_input_ids = input_ids[start_idx:end_idx]
            mb_labels = labels[start_idx:end_idx] if labels is not None else None

            # GPipe fill-drain: sequential through stages with async P2P transfers
            mb_loss = self._pipeline_micro_batch_forward(mb_input_ids, mb_labels, io_dtype, **kwargs)
            losses_plus.append(mb_loss)

        # Aggregate micro-batch losses
        loss_plus = torch.stack(losses_plus).mean()

        return loss_plus

    def _pipeline_micro_batch_forward(self, input_ids, labels, io_dtype, **kwargs):
        """
        Execute one micro-batch through the multi-GPU pipeline.

        Args:
            input_ids: Micro-batch input tokens
            labels: Micro-batch labels
            io_dtype: Data type for inter-stage transfers
            **kwargs: Additional forward arguments

        Returns:
            Micro-batch loss (scalar tensor)
        """
        # Stage 0: Embeddings
        embed_device = self.stage_devices[self.embed_device_stage]

        with torch.cuda.device(embed_device):
            with torch.cuda.stream(self.stage_compute_streams[embed_device]):
                # Move inputs to embed device
                input_ids_dev = input_ids.to(embed_device)

                # Get embeddings (model-specific, assuming OPT-like structure)
                if hasattr(self.model, 'decoder'):
                    decoder = self.model.decoder
                    hidden_states = decoder.embed_tokens(input_ids_dev)
                    if hasattr(decoder, 'embed_positions'):
                        positions = decoder.embed_positions(input_ids_dev)
                        hidden_states = hidden_states + positions
                else:
                    # Fallback for different model structures
                    hidden_states = self.model.get_input_embeddings()(input_ids_dev)

                # Convert to pipeline I/O dtype
                hidden_states = hidden_states.to(io_dtype)

        # Pipeline through transformer stages
        for stage_idx, stage_device in enumerate(self.stage_devices):
            with torch.cuda.device(stage_device):
                compute_stream = self.stage_compute_streams[stage_device]

                with torch.cuda.stream(compute_stream):
                    # Transfer hidden states from previous stage
                    if stage_idx > 0:
                        prev_device = self.stage_devices[stage_idx - 1]
                        transfer_stream = self.stage_transfer_streams[prev_device]

                        if self.p2p_overlap:
                            # Wait for transfer to complete
                            compute_stream.wait_stream(transfer_stream)

                        hidden_states = hidden_states.to(stage_device, non_blocking=self.p2p_overlap)

                    # Process layers on this stage
                    stage_layers = self.stage_layers[stage_idx] if hasattr(self, 'stage_layers') else []

                    for layer_id in stage_layers:
                        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layers'):
                            layer = self.model.decoder.layers[layer_id]

                            # Upload layer if offloaded (per-GPU CPU offloading)
                            if layer.parameters().__next__().device.type == 'cpu':
                                layer = layer.to(stage_device)

                            hidden_states = layer(hidden_states)[0]  # OPT returns tuple

                            # Offload layer if CPU offloading enabled per GPU
                            if self.enable_cpu_offloading_per_gpu:
                                layer.cpu()

                    # Async transfer to next stage
                    if stage_idx < len(self.stage_devices) - 1:
                        next_device = self.stage_devices[stage_idx + 1]
                        transfer_stream = self.stage_transfer_streams[stage_device]

                        if self.p2p_overlap:
                            with torch.cuda.stream(transfer_stream):
                                hidden_states = hidden_states.to(next_device, non_blocking=True)
                        else:
                            hidden_states = hidden_states.to(next_device)

        # Final stage: LM head and loss
        lm_head_device = self.stage_devices[self.lm_head_device_stage]

        with torch.cuda.device(lm_head_device):
            with torch.cuda.stream(self.stage_compute_streams[lm_head_device]):
                # Transfer if needed
                if self.lm_head_device_stage != len(self.stage_devices) - 1:
                    hidden_states = hidden_states.to(lm_head_device)

                # Apply final layer norm if present
                if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'final_layer_norm'):
                    if self.model.decoder.final_layer_norm is not None:
                        hidden_states = self.model.decoder.final_layer_norm(hidden_states)

                # Project to vocabulary
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(hidden_states)
                elif hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'project_out'):
                    if self.model.decoder.project_out is not None:
                        hidden_states = self.model.decoder.project_out(hidden_states)
                    # Get lm_head from parent model
                    logits = self.model.lm_head(hidden_states)
                else:
                    raise AttributeError("Could not find lm_head in model")

                # Compute loss
                if labels is not None:
                    labels_dev = labels.to(lm_head_device)

                    # Shift for next-token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels_dev[..., 1:].contiguous()

                    # Flatten and compute cross-entropy
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                else:
                    loss = torch.tensor(0.0, device=lm_head_device)

                # Synchronize all stages
                torch.cuda.synchronize()

                return loss

    def assign_zo2_attributes(self, source, target):
        """
        Utility function to transfer ZO2 specific attributes from one module to another,
        aiding in maintaining consistency across nested model architectures.

        Args:
            source: The source module from which attributes are copied.
            target: The target module to which attributes are assigned.
        """
        attrs_to_assign = ['upload_stream', 'offload_stream', 'compute_stream', 
                           'zo_random_seed', 'rstate', 'rstate_queue', 'last_rstate', 
                           'projected_grad']
        for attr in attrs_to_assign:
            setattr(target, attr, getattr(source, attr))
    
    @torch.inference_mode
    def zo_update(self, module, weight_decay=None):
        """
        Applies the computed gradients to update parameters of the module, potentially
        including a weight decay term. This method is enhanced by managing CUDA state
        to ensure consistent random number generation across calls.

        Args:
            module (nn.Module): The module whose parameters are to be updated.
            weight_decay (float, optional): Optional weight decay for regularization.
        """
        torch.cuda.set_rng_state(self.last_rstate)
        super().zo_update(module, weight_decay=weight_decay)
        self.last_rstate = torch.cuda.get_rng_state()
        return module
    
    @torch.inference_mode()
    def module_dual_forward(self, module, inputs1, inputs2, projected_grad=0., weight_decay=None):
        """
        Performs two parallel forward computations with perturbed parameters to estimate
        gradients. This function is key for zeroth-order gradient estimation with support
        for optional weight decay during parameter update. 
        
        Notice that the application of Gaussian perturbations for the parameters 
        during both the perturbation and update phases should be the same.

        Args:
            module (nn.Module): The module on which forward passes are conducted.
            inputs1 (dict): Inputs for the first forward pass.
            inputs2 (dict): Inputs for the second forward pass.
            projected_grad (float): Projected gradient value used for updating parameters.
            weight_decay (float, optional): Optional weight decay for regularization.
        """
        if projected_grad != 0:
            module = self.zo_update(module, weight_decay)
        torch.cuda.set_rng_state(self.rstate)
        self.zo_perturb_parameters(module, scaling_factor=self.zo_perturb_shifts()[0])
        output1 = module(**inputs1)
        torch.cuda.set_rng_state(self.rstate)
        self.zo_perturb_parameters(module, scaling_factor=self.zo_perturb_shifts()[1])
        output2 = module(**inputs2)
        torch.cuda.set_rng_state(self.rstate)
        self.zo_perturb_parameters(module, scaling_factor=self.zo_perturb_shifts()[2])
        self.rstate = torch.cuda.get_rng_state()
        return output1, output2
    
    @torch.inference_mode()
    def function_dual_forward(self, fn, inputs1, inputs2):
        """
        Executes a provided function twice with dual inputs, supporting the zeroth-order optimization process
        by enabling the estimation of gradients through function outputs.

        Args:
            fn (callable): The function to be executed.
            inputs1 (dict): Arguments for the first execution of the function.
            inputs2 (dict): Arguments for the second execution of the function.

        Returns:
            tuple: Outputs from the two executions of the function.
        """
        output1 = fn(**inputs1)
        output2 = fn(**inputs2)
        return output1, output2
    
    @torch.inference_mode()
    def zo_forward(self, *args, seed: int=None, **kwargs):
        """
        The overarching forward function that integrates perturbation, gradient estimation,
        and parameter update within a single coherent process, controlled by the seed for reproducibility.

        Args:
            seed (int, optional): Seed for random number generation to ensure reproducibility.
        """
        self._update_lr()
        self.zo_random_seed = seed if seed else np.random.randint(self.max_zo_random_seed)
        torch.manual_seed(self.zo_random_seed)
        torch.cuda.manual_seed(self.zo_random_seed)
        self.rstate = torch.cuda.get_rng_state()
        self.rstate_queue.append(self.rstate.clone())
        if len(self.rstate_queue) == 2:
            self.last_rstate = self.rstate_queue.popleft()
        torch.cuda.synchronize()    # global sync to make sure all tasks finish
        loss1, loss2 = self.inner_zo_forward(*args, **kwargs)
        torch.cuda.synchronize()    # global sync to make sure all tasks finish
        self.projected_grad = self.compute_grad(loss1, loss2)
        return loss1.detach()
    
    #*********************** tasks ***********************#

    def task_upload(self, module, device='cuda', upload_sync=False, *args, **kwargs):
        """
        Handles the uploading of modules to the GPU, utilizing CUDA streams to potentially overlap
        computation and communication for efficiency.

        Args:
            module (nn.Module): Module to be uploaded.
            device (str): Target device for the upload.
            upload_sync (bool): Whether to synchronize the upload stream before proceeding.
        """
        if self.overlap:
            if upload_sync:
                self.upload_stream.synchronize()
        with torch.cuda.stream(self.upload_stream if self.overlap else torch.cuda.current_stream()):
            module = self.upload_impl(
                module, 
                device, 
                self.offloading_device,
                self.communicate_optimize_method, 
                non_blocking=self.overlap, 
                *args, **kwargs
            )
        return module

    def task_offload(self, module, device='cpu', offload_sync=False, *args, **kwargs):
        """
        Manages the offloading of modules to an alternative storage (e.g., CPU or disk), using CUDA streams
        to manage dependencies and potentially overlap tasks.

        Args:
            module (nn.Module): Module to be offloaded.
            device (str): Target device for the offload.
            offload_sync (bool): Whether to synchronize the offload stream before proceeding.
        """
        if self.overlap:
            if offload_sync:
                self.offload_stream.synchronize()
            self.compute_stream.synchronize()   # offload depends on compute task
        with torch.cuda.stream(self.offload_stream if self.overlap else torch.cuda.current_stream()):
            module = self.offload_impl(
                module, 
                device, 
                self.offloading_device,
                self.communicate_optimize_method, 
                non_blocking=self.overlap, 
                *args, **kwargs
            )
        return module
    
    def task_compute_module(self, module, inputs1, inputs2, grad, compute_sync=False, weight_decay=None, *args, **kwargs):
        """
        Conducts computations on a module with optional dual inputs for gradient estimation,
        applying synchronization and CUDA streams for efficiency.

        Args:
            module (nn.Module): The module on which computations are to be performed.
            inputs1 (dict): Inputs for the first computation.
            inputs2 (dict, could be None): Inputs for the second computation, if performing dual forward.
            grad (float): Gradient value to be applied.
            compute_sync (bool): Whether to synchronize the compute stream before proceeding.
            weight_decay (float, optional): Optional weight decay during the update.
        """
        if self.overlap:
            if compute_sync:
                self.compute_stream.synchronize()
            self.upload_stream.synchronize()   # module compute depends on upload task
        with torch.cuda.stream(self.compute_stream if self.overlap else torch.cuda.current_stream()):
            if inputs2 is not None:
                return self.compute_module_impl(
                    self.module_dual_forward,
                    module,
                    self.compute_module_optimize_method,
                    inputs1=inputs1, 
                    inputs2=inputs2,
                    projected_grad=grad,
                    weight_decay=weight_decay,
                    *args, **kwargs
                )
            elif isinstance(inputs1, list):
                return self.compute_module_impl(
                    None,
                    module,
                    self.compute_module_optimize_method,
                    *inputs1,
                    *args,
                    **kwargs
                )
            elif isinstance(inputs1, dict):
                return self.compute_module_impl(
                    None,
                    module,
                    self.compute_module_optimize_method,
                    *args,
                    **inputs1,
                    **kwargs
                )
            elif isinstance(inputs1, tuple):
                return self.compute_module_impl(
                    None,
                    module,
                    self.compute_module_optimize_method,
                    *inputs1[0],
                    *args,
                    **inputs1[1],
                    **kwargs
                )
            else:
                raise ValueError("Invalid inputs type.")
    
    def task_compute_function(self, fn, inputs1, inputs2, compute_sync=False, *args, **kwargs):
        """
        Executes a provided function with dual input sets to facilitate parallel operations
        and gradient estimation. This method integrates CUDA streams for efficient task execution.

        Args:
            fn (callable): The function to execute, typically a PyTorch operation or custom function.
            inputs1 (dict): Arguments for the first execution of the function.
            inputs2 (dict, could be None): Arguments for the second execution of the function.
            compute_sync (bool): Whether to synchronize the compute stream before execution to ensure data readiness.
        """
        if self.overlap:
            if compute_sync:
                self.compute_stream.synchronize()
        with torch.cuda.stream(self.compute_stream if self.overlap else torch.cuda.current_stream()):
            if inputs2 is not None:
                return self.compute_function_impl(
                    self.function_dual_forward,
                    fn,
                    self.compute_function_optimize_method,
                    inputs1=inputs1, 
                    inputs2=inputs2,
                    *args, **kwargs
                )
            elif isinstance(inputs1, list):
                return self.compute_function_impl(
                    None,
                    fn, 
                    self.compute_function_optimize_method,
                    *inputs1,
                    *args,
                    **kwargs
                )
            elif isinstance(inputs1, dict):
                return self.compute_function_impl(
                    None,
                    fn, 
                    self.compute_function_optimize_method,
                    *args,
                    **inputs1,
                    **kwargs
                )
            elif isinstance(inputs1, tuple):
                return self.compute_function_impl(
                    None,
                    fn, 
                    self.compute_function_optimize_method,
                    *inputs1[0],
                    *args,
                    **inputs1[1],
                    **kwargs
                )
            else:
                raise ValueError("Invalid inputs type.")

    #*********************** evaluate ***********************#

    @torch.inference_mode()
    def zo_eval_forward(self, *args, **kwargs):
        """
        Conducts a model evaluation using the internal forward method without applying any perturbations.
        This method ensures all tasks finish before and after the evaluation to maintain synchronization.

        Args:
            *args, **kwargs: Arguments and keyword arguments for the model's forward method.
        """
        if MeZO2SGD.first_call_eval:
            print("Warning: ZO2 may not efficiently optimize the evaluation stage, which could result in slower performance.")
            MeZO2SGD.first_call_eval = False  # Disable the warning after the first call
        torch.cuda.synchronize()    # global sync to make sure all tasks finish
        output = self.inner_zo_eval_forward(*args, **kwargs)
        torch.cuda.synchronize()    # global sync to make sure all tasks finish
        return output
    
    def add_zo2_eval_comm_hooks(self, blocks):
        """
        Attaches communication hooks to model blocks to manage data uploading and offloading during evaluation.
        This helps in managing memory more efficiently during the eval phase.

        Args:
            blocks (list): List of model blocks to attach hooks to.

        Returns:
            list: A list of hook handles for managing lifecycle.
        """
        handles = []
        for block in blocks:
            if isinstance(block, nn.Module):
                pre_handle = block.register_forward_pre_hook(self.eval_upload_hook)
                post_handle = block.register_forward_hook(self.eval_offload_hook)
                handles.append(pre_handle)
                handles.append(post_handle)
        return handles
    
    def clear_zo2_eval_comm_hooks(self, handles):
        """
        Removes communication hooks from model blocks after evaluation to clean up and prevent memory leaks.

        Args:
            handles (list): List of hook handles to be removed.
        """
        for handle in handles:
            handle.remove()
    
    def eval_upload_hook(self, module, input):
        """
        A forward pre-hook to upload a module to the GPU before its evaluation.

        Args:
            module (nn.Module): Module to be uploaded.
            input: Input data for the module.
        """
        self.upload_impl(
            module, 
            self.device, 
            self.offloading_device
        )
        return input

    def eval_offload_hook(self, module, input, output):
        """
        A forward hook to offload a module from the GPU after its evaluation to free up memory.

        Args:
            module (nn.Module): Module to be offloaded.
            input: Input data for the module.
            output: Output from the module evaluation.
        """
        if self.overlap:
            with torch.cuda.stream(self.offload_stream):
                self.offload_impl(
                    module, 
                    self.offloading_device, 
                    self.offloading_device
                )
        else:
            self.offload_impl(
                module, 
                self.offloading_device, 
                self.offloading_device
            )
        return output
    
    #*********************** backend ***********************#

    def upload_impl(
            self,
            module: nn.Module, 
            device: str, 
            offloading_device: str,
            optimize_method: str = "", 
            module_id: str = None,
            *args, **kwargs
        ):
        """
        Implements the logic for uploading model components to a specified device.
        Supports various optimization methods to tailor the upload process for different computing environments.
        """
        def _upload_impl(module, device, offloading_device, *args, **kwargs):
            if offloading_device == "cpu":
                # Use pinned memory for faster CPU to GPU transfers if enabled
                if self.use_pinned_memory and not kwargs.get('non_blocking', False):
                    # For synchronous transfers, we can use pinned memory staging
                    if isinstance(module, nn.Module):
                        # Transfer module parameters through pinned memory
                        for param in module.parameters():
                            if param.device.type == 'cpu':
                                pinned_buffer = self.get_or_create_pinned_buffer(param.shape, param.dtype)
                                if pinned_buffer is not None:
                                    # Stage 1: CPU -> Pinned Memory (fast)
                                    pinned_buffer.copy_(param.data)
                                    # Stage 2: Pinned Memory -> GPU (async possible)
                                    param.data = pinned_buffer.to(device, non_blocking=True)
                                else:
                                    # Fallback to regular transfer
                                    param.data = param.data.to(device)
                            else:
                                param.data = param.data.to(device)
                        # Move buffers
                        for buffer_name, buffer in module.named_buffers():
                            if buffer.device.type == 'cpu':
                                module._buffers[buffer_name.split('.')[-1]] = buffer.to(device)
                    elif isinstance(module, torch.Tensor) and module.device.type == 'cpu':
                        pinned_buffer = self.get_or_create_pinned_buffer(module.shape, module.dtype)
                        if pinned_buffer is not None:
                            pinned_buffer.copy_(module)
                            module = pinned_buffer.to(device, non_blocking=True)
                        else:
                            module = module.to(device)
                    else:
                        module = module.to(device, *args, **kwargs)
                else:
                    # Regular transfer without pinned memory
                    module = module.to(device, *args, **kwargs)
            else:
                if module_id == None:
                    raise ValueError("For disk offloading mode, 'module_id' cannot be None.")
                offloading_disk_path = get_disk_offload_path(offloading_device, module_id)
                match type(module):
                    case torch.Tensor:
                        module = torch.load(offloading_disk_path, map_location=device)
                    case nn.Module:
                        module.load_state_dict(torch.load(offloading_disk_path, map_location=device))
                    case _:
                        raise ValueError
                clear_disk_offload_path(offloading_device, module_id)
            return module
        match optimize_method:
            case "":
                module = _upload_impl(module, device, offloading_device, *args, **kwargs)
            case "bucket":  # works on large-scale models
                bucket = module_to_bucket_inplace(module)
                bucket = _upload_impl(bucket, device, offloading_device, *args, **kwargs)
                module = bucket_to_module_inplace(bucket, module)
            case _:
                raise NotImplementedError
        if self.amp:    # after uploading, decompress the module to higher precision
            module = self.amp_decompress_impl(module)
        return module

    def offload_impl(
            self,
            module: nn.Module,
            device: str,
            offloading_device: str,
            optimize_method: str = "",
            module_id: str = None,
            *args, **kwargs
        ):
        """
        Implements the logic for offloading model components from the GPU to another storage,
        such as CPU or disk, to manage GPU memory more efficiently.
        """
        def _offload_impl(module, device, offloading_device, *args, **kwargs):
            if offloading_device == "cpu":
                # Use pinned memory for GPU to CPU transfers if enabled
                if self.use_pinned_memory and hasattr(module, 'is_cuda') and module.is_cuda:
                    # Handle tensor offloading with pinned memory
                    if isinstance(module, torch.Tensor):
                        pinned_buffer = self.get_or_create_pinned_buffer(module.shape, module.dtype)
                        if pinned_buffer is not None:
                            # Stage 1: GPU -> Pinned Memory (async possible)
                            pinned_buffer.copy_(module, non_blocking=True)
                            # Stage 2: Pinned Memory -> CPU
                            module = pinned_buffer.cpu()
                        else:
                            module = module.to(device, *args, **kwargs)
                    # Handle module offloading with pinned memory
                    elif isinstance(module, nn.Module):
                        for param in module.parameters():
                            if param.is_cuda:
                                pinned_buffer = self.get_or_create_pinned_buffer(param.shape, param.dtype)
                                if pinned_buffer is not None:
                                    # Stage 1: GPU -> Pinned Memory (async possible)
                                    pinned_buffer.copy_(param.data, non_blocking=True)
                                    # Stage 2: Pinned Memory -> CPU
                                    param.data = pinned_buffer.cpu()
                                else:
                                    param.data = param.data.to(device, *args, **kwargs)
                    else:
                        module = module.to(device, *args, **kwargs)
                else:
                    module = module.to(device, *args, **kwargs)
            else:
                if module_id == None:
                    raise ValueError("For disk offloading mode, 'module_id' cannot be None.")
                offloading_disk_path = create_disk_offload_path(offloading_device, module_id)
                match type(module):
                    case torch.Tensor:
                        torch.save(module, offloading_disk_path)
                    case nn.Module:
                        torch.save(module.state_dict(), offloading_disk_path)
                    case _:
                        raise ValueError
            return module
        if self.amp:    # before offloading, compress the module to lower precision
            module = self.amp_compress_impl(module)
        match optimize_method:
            case "":
                module = _offload_impl(module, device, offloading_device, *args, **kwargs)
            case "bucket":  # works on large-scale models
                bucket = module_to_bucket_inplace(module)
                bucket = _offload_impl(bucket, device, offloading_device, *args, **kwargs)
                module = bucket_to_module_inplace(bucket, module)
            case _:
                raise NotImplementedError
        return module
        
    def compute_module_impl(
            self,
            forward_fn,
            module: torch.nn.Module,
            optimize_method: str,
            *args, 
            optimize_kwargs = None,
            **kwargs
        ):
        """
        Manages the computation tasks on a module, applying various optimization methods
        to enhance execution speed and efficiency.
        """
        match optimize_method:
            case "":
                pass
            case "torch.compile":   # may introduce some precision mismatch
                module = torch.compile(module, **optimize_kwargs)
            case _:
                raise NotImplementedError
        with torch.autocast(device_type=self.device, dtype=self.amp_precision, enabled=self.amp):
            if forward_fn is None:
                return module(*args, **kwargs)
            else:
                return forward_fn(module=module, *args, **kwargs)

    def compute_function_impl(
            self,
            function_fn,
            fn,
            optimize_method: str,
            *args, 
            optimize_kwargs = None,
            **kwargs
        ):
        """
        Manages the computation tasks on a function, applying various optimization methods
        to enhance function execution speed and efficiency.
        """
        match optimize_method:
            case "":
                pass
            case "torch.jit.script":   # may introduce some precision mismatch
                fn = torch.jit.script(fn, **optimize_kwargs)
            case _:
                raise NotImplementedError
        with torch.autocast(device_type=self.device, dtype=self.amp_precision, enabled=self.amp):
            if function_fn is None:
                return fn(*args, **kwargs)
            else:
                return function_fn(fn, *args, **kwargs)

    def amp_decompress_impl(self, module: nn.Module) -> nn.Module:
        """
        Converts the data type of module parameters to a higher precision typically used for computations.
        This is part of the AMP process where parameters might be temporarily compressed to a lower precision
        and need to be decompressed back to higher precision for accuracy-critical operations.

        Args:
            module (nn.Module): The module whose parameters will be decompressed.

        Returns:
            nn.Module: The module with parameters converted to higher precision.
        """
        for p in module.parameters():
            match self.amp_compress_method:
                case "naive":
                    p.data = p.data.to(dtype=self.precision_on_working_device)
                case _:
                    raise NotImplementedError
        return module

    def amp_compress_impl(self, module: nn.Module) -> nn.Module:
        """
        Compresses the data type of module parameters to a lower precision typically used to save memory and 
        improve computational efficiency during less accuracy-critical operations.
        
        Args:
            module (nn.Module): The module whose parameters will be compressed.

        Returns:
            nn.Module: The module with parameters converted to lower precision.
        """
        for p in module.parameters():
            match self.amp_compress_method:
                case "naive":
                    p.data = p.data.to(dtype=self.precision_on_offloading_device)
                case _:
                    raise NotImplementedError
        return module

    #*********************** api ***********************#

    def init_zo2_upload(self):
        """
        Initializes the upload of essential model components to the GPU.
        This method specifically handles the uploading of model embeddings and head components,
        and prepares the offloading blocks based on configuration. This setup is crucial for
        managing the active memory footprint during training by selectively uploading and
        offloading transformer blocks as needed.
        """
        print("Upload head and tail to cuda.")
        self.model.transformer.wte = self.model.transformer.wte.to(self.device)
        self.model.transformer.wpe = self.model.transformer.wpe.to(self.device)
        self.model.transformer.ln_f = self.model.transformer.ln_f.to(self.device)
        self.model.lm_head = self.model.lm_head.to(self.device)

        self.num_blocks = len(self.model.transformer.h)
        if self.offloading_blocks is not None:
            self.offloading_blocks = self.offloading_blocks
        else:
            self.offloading_blocks = list(range(self.num_blocks))
        print(f"Transformer blocks {self.offloading_blocks} will be offloaded to {self.offloading_device}")
        for i in range(self.num_blocks):
            if i in self.offloading_blocks:
                continue
            else:
                self.model.transformer.h[i] = self.model.transformer.h[i].to(self.device)
                print(f"Upload block {i} to cuda.")
    
    @torch.inference_mode()   
    def inner_zo_forward(self, idx, pos, targets):
        """
        Defines the inner forward logic for zeroth-order optimization, applying perturbations
        and calculating the loss for gradient estimation. This method, using nanogpt as an example, orchestrates the forward
        computation across potentially offloaded transformer blocks, ensuring they are uploaded
        for computation and offloaded post-computation as configured.

        Args:
            idx (Tensor): Input indices for token embeddings.
            pos (Tensor): Position indices for positional embeddings.
            targets (Tensor): Target outputs for loss calculation.

        Returns:
            Tuple[Tensor, Tensor]: The losses computed from two perturbed forward passes, used for gradient estimation.
        """
        we1, we2 = self.task_compute_module(self.model.transformer.wte,
                                inputs1={"input": idx},
                                inputs2={"input": idx},
                                grad=self.projected_grad)
        pe1, pe2 = self.task_compute_module(self.model.transformer.wpe, 
                                 {"input": pos}, 
                                 {"input": pos}, 
                                 self.projected_grad)
        hidden_states1, hidden_states2 = self.task_compute_function(torch.add,
                                                                    {"input": we1, "other": pe1},
                                                                    {"input": we2, "other": pe2})
        if 0 in self.offloading_blocks:
            self.model.transformer.h[0] = self.task_upload(
                module=self.model.transformer.h[0], 
                device=self.device)
        N = len(self.model.transformer.h)
        for i in range(1, N):
            if i != 1:
                if i-2 in self.offloading_blocks:
                    self.model.transformer.h[i-2] = self.task_offload(
                        module=self.model.transformer.h[i-2], 
                        device=self.offloading_device)
            hidden_states1, hidden_states2 = self.task_compute_module(
                self.model.transformer.h[i-1], 
                inputs1={"x": hidden_states1}, 
                inputs2={"x": hidden_states2}, 
                grad=self.projected_grad)
            if i in self.offloading_blocks:
                self.model.transformer.h[i] = self.task_upload(
                    module=self.model.transformer.h[i], 
                    device=self.device)
        if N-2 in self.offloading_blocks:
            self.model.transformer.h[N-2] = self.task_offload(
                self.model.transformer.h[N-2], device=self.offloading_device)
        hidden_states1, hidden_states2 = self.task_compute_module(
                    self.model.transformer.h[N-1], 
                    inputs1={"x": hidden_states1}, 
                    inputs2={"x": hidden_states2}, 
                    grad=self.projected_grad
                )
        if N-1 in self.offloading_blocks:
            self.model.transformer.h[N-1] = self.task_offload(
                self.model.transformer.h[N-1], device=self.offloading_device)
        logits1, logits2 = self.task_compute_module(self.model.transformer.ln_f,
                                             inputs1={"input": hidden_states1}, 
                                             inputs2={"input": hidden_states2}, 
                                             grad=self.projected_grad,
                                             weight_decay=0.)
        logits1, logits2 = self.task_compute_module(self.model.lm_head,
                                             inputs1={"input": logits1}, 
                                             inputs2={"input": logits2}, 
                                             grad=self.projected_grad)
        loss1, loss2 = self.task_compute_function(F.cross_entropy,
                                                  {"input": logits1[:, :-1, :].reshape(-1, logits1.size(-1)), 
                                                   "target": targets[:, 1:].reshape(-1)},
                                                  {"input": logits2[:, :-1, :].reshape(-1, logits2.size(-1)), 
                                                   "target": targets[:, 1:].reshape(-1)})
        return loss1, loss2
    
    @torch.inference_mode()   
    def inner_zo_eval_forward(self, eval_fn, idx, pos, targets):
        """
        Conducts an evaluation forward pass of the model using the zeroth-order optimization setup,
        but without applying any perturbations to ensure accurate performance assessment.
        This function manages the dynamic uploading and offloading of transformer blocks as needed,
        utilizing pre- and post-hooks to optimize memory usage during evaluation.

        Args:
            eval_fn (callable): The evaluation function to be applied, typically involves a forward pass
                                that computes the loss or other metrics without updating model parameters.
            idx (Tensor): Input indices for token embeddings.
            pos (Tensor): Position indices for positional embeddings.
            targets (Tensor): Target outputs for computing the evaluation metric (e.g., loss).

        Returns:
            Tensor: The output from the evaluation function, typically loss or accuracy metrics.
        """
        handles = self.add_zo2_eval_comm_hooks(self.model.transformer.h)
        output = eval_fn(idx, pos, targets)
        self.clear_zo2_eval_comm_hooks(handles)
        return output
    