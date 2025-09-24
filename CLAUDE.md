# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZO2 (Zeroth-Order Offloading) is a framework for full parameter fine-tuning of large language models (LLMs) with limited GPU memory. It enables fine-tuning models up to 175B parameters with as little as 18GB GPU memory using zeroth-order optimization and CPU offloading techniques.

Track task backlog via notion tool in notion database ZO2 AI Agent Investigation Backlog. Update task progress and results as sub page of corresponding task in notion markdown format.

## GPU Environment Access

### SSH Key Authentication (Configured)
SSH key authentication is now configured for passwordless access:

```bash
# Direct connection
ssh -p 13727 root@rent-gpus.netmind.ai

# Using convenient alias (recommended)
ssh gpu-server

# Working directory
cd /root/lei/zo2
```

**Note**: SSH key authentication has been set up using the local `~/.ssh/id_rsa.pub` key. No password needed!

### SSH Configuration
The following aliases are configured in `~/.ssh/config`:
- `gpu-server`: Alias for quick access to rent-gpus.netmind.ai
- `rent-gpus.netmind.ai`: Direct hostname with proper configuration

### SSH Tools Available
The following SSH utilities are installed on the local macOS machine:
- **autossh**: Automatically restart SSH sessions (`/opt/homebrew/bin/autossh`)
- **mosh**: Mobile shell with roaming and intermittent connectivity (`/opt/homebrew/bin/mosh`)
- **ssh-copy-id**: Install SSH keys on remote servers (`/opt/homebrew/opt/ssh-copy-id/bin/ssh-copy-id`)

### Legacy Password Access (Not Recommended)
If SSH key authentication fails for any reason:
- Password: `Test_2024`
- Can use `sshpass -p "Test_2024" ssh -p 13727 root@rent-gpus.netmind.ai` (not recommended)

### System Resources
- **GPUs**: 4x NVIDIA GeForce RTX 3090 (24GB VRAM each)
- **OS**: Ubuntu
- **RAM**: 251GB total
- **Storage**:
  - `/` (overlay): 3.0TB total, 2.0TB available - **Use for large models and datasets**
  - `/root/.env` (nvme0n1p2): 1.8TB total, 662GB available - Fast NVMe storage
  - `/etc/hosts` (sda1): 3.7TB total, 1.9TB available - Additional storage
  
**Recommendation**: 
- Store large models and datasets in the root filesystem (`/`) which has 2.0TB available space, or use `/root/lei/` for persistent project data.
- Create conda virtual environment before installing dependancies. 

## Commands

### Setup and Installation

```bash
# Create conda environment from yml file
conda env create -f env.yml
conda activate zo2

# Install as package
pip install git+https://github.com/liangyuwang/zo2.git
```

### Running Tests

```bash
# Test memory usage across different OPT model sizes
bash test/mezo_sgd/hf_opt/record_zo2_memory.sh

# Compare memory usage
bash test/mezo_sgd/hf_opt/test_memory_train.sh

# Test throughput/speed
bash test/mezo_sgd/hf_opt/test_speed_train.sh

# Test accuracy
bash test/mezo_sgd/hf_opt/test_acc_train.sh
```

### Running MeZO Examples

```bash
# Navigate to example directory
cd example/mezo_runner/

# Run OPT-2.7B fine-tuning
MODEL=facebook/opt-2.7b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 STEPS=20000 EVAL_STEPS=4000 bash mezo.sh

# Run Qwen3-1.7B fine-tuning  
MODEL=Qwen/Qwen3-1.7B TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 STEPS=20000 EVAL_STEPS=4000 bash mezo.sh
```

## Architecture

### Core Components

**zo2/** - Main package directory containing:
- `config/` - Configuration classes for different ZO methods (MeZOSGDConfig)
- `model/` - Model implementations for NanoGPT and HuggingFace transformers (OPT, Qwen3)
- `optimizer/` - ZO optimizer implementations including MeZO-SGD with CPU offloading
- `trainer/` - Training utilities for HuggingFace integration (ZOTrainer, ZOSFTTrainer)
- `utils/` - Helper utilities

### Key Design Patterns

1. **ZO Configuration**: All ZO methods are configured through `ZOConfig` factory that returns appropriate config objects
2. **Model Initialization**: Models must be initialized within `zo_hf_init` context manager and call `model.zo_init(zo_config)`
3. **Offloading Strategy**: Uses dynamic scheduling to optimize computation-communication overlap between CPU and GPU
4. **Training Modes**: Supports both custom training loops and HuggingFace Trainer integration

### Supported Models and Tasks

**Models**:
- NanoGPT (for testing)
- HuggingFace OPT models (125M to 175B parameters)
- Qwen3 models (including 32B FP8 version)

**Tasks** (via MeZO-Runner):
- SST2, Copa, BoolQ, MultiRC, CB, WIC, WSC, ReCoRD, RTE, SQuAD, DROP

## Development Notes

- Python 3.11+ required
- Dependencies: PyTorch 2.4.0+, CUDA 12.1+, transformers 4.51.3
- For OPT-175B: Requires 18GB GPU memory and 600GB CPU memory
- Use `--train_as_classification` flag for classification tasks except Copa, ReCoRD, SQuAD, DROP