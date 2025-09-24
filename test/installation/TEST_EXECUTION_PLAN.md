# ZO2 Phase 2: Test Execution Plan

## Overview
This document provides a comprehensive plan for executing Phase 2 Build & Installation tests for the ZO2 framework.

## Test Environment Setup

### Prerequisites
- Python 3.11+
- CUDA 12.1+ (for GPU testing)
- 4x NVIDIA RTX 3090 GPUs (or similar)
- 251GB RAM
- Ubuntu OS (on GPU server)

### SSH Access to GPU Server
```bash
# Direct connection (passwordless with SSH key)
ssh gpu-server

# Navigate to project directory
cd /root/lei/zo2
```

## Test Execution Phases

### Phase 2.1: Installation Method Testing

#### Test 2.1.1: Conda Environment Installation
```bash
# Clean previous installation
conda deactivate
conda env remove -n zo2_test -y

# Test conda installation
conda env create -f env.yml -n zo2_test
conda activate zo2_test

# Verify installation
python test/installation/quick_test.py

# Expected Output:
# - All imports successful
# - CUDA available with 4 GPUs
# - ZO2 modules loaded
```

#### Test 2.1.2: Pip Installation from GitHub
```bash
# Create fresh environment
conda create -n zo2_pip python=3.11 -y
conda activate zo2_pip

# Install from GitHub
pip install git+https://github.com/liangyuwang/zo2.git

# Verify installation
python test/installation/quick_test.py

# Expected Output:
# - Package installed without conflicts
# - All dependencies resolved
# - Core functionality available
```

### Phase 2.2: Compatibility Testing

#### Test 2.2.1: Module Import Verification
```bash
python test/installation/test_imports.py

# Expected Output:
# - All 15+ ZO2 modules import successfully
# - No missing dependencies
# - Summary: 15/15 modules imported
```

#### Test 2.2.2: PyTorch Compatibility
```bash
python test/installation/test_pytorch_version.py

# Expected Output:
# - PyTorch version >= 2.4.0
# - All required components available
# - Mixed precision support confirmed
# - Memory management features available
```

#### Test 2.2.3: CUDA Compatibility
```bash
python test/installation/test_cuda_compatibility.py

# Expected Output:
# - CUDA version >= 12.1
# - 4 GPUs detected (24GB each)
# - CUDA operations working
# - Memory allocation successful
```

#### Test 2.2.4: Transformers Compatibility
```bash
python test/installation/test_transformers_compat.py

# Expected Output:
# - Transformers version >= 4.51.3
# - OPT models available
# - Training utilities functional
# - Optional: Qwen models available
```

### Phase 2.3: Functional Testing

#### Test 2.3.1: Minimal Training Test
```bash
python test/installation/test_minimal_training.py

# Expected Output:
# - Model initialization successful
# - 3 training steps completed
# - Loss values reasonable (0 < loss < 100)
# - Memory usage < 1GB for tiny model
# - Evaluation mode working
```

#### Test 2.3.2: Memory Usage Validation
```bash
# Test OPT-125M model
cd /root/lei/zo2
python test/mezo_sgd/hf_opt/test_memory.py \
    --model_name opt_125m \
    --task causalLM \
    --zo_method zo2 \
    --max_steps 10

# Expected Output:
# - Peak GPU Memory < 2GB
# - Peak CPU Memory reported
# - No OOM errors
```

#### Test 2.3.3: Speed/Throughput Test
```bash
python test/mezo_sgd/hf_opt/test_speed.py \
    --model_name opt_125m \
    --task causalLM \
    --zo_method zo2 \
    --max_steps 10

# Expected Output:
# - Throughput in tokens/second
# - No performance degradation
# - Stable iteration times
```

### Phase 2.4: Comprehensive Test Suite

#### Run All Tests
```bash
cd /root/lei/zo2
bash test/installation/run_all_tests.sh

# Expected Output:
# - Test Suite 1: Module Imports - PASSED
# - Test Suite 2: PyTorch Compatibility - PASSED
# - Test Suite 3: CUDA Compatibility - PASSED
# - Test Suite 4: Transformers Compatibility - PASSED
# - Test Suite 5: Functional Tests - PASSED
# - Summary: 5/5 test suites passed
```

## Test Validation Criteria

### PASS Criteria
- All core modules import without errors
- PyTorch version >= 2.4.0
- CUDA version >= 12.1
- Transformers version >= 4.51.3
- Minimal training completes without errors
- Memory usage within expected bounds
- No dependency conflicts

### FAIL Criteria
- Any core module fails to import
- Version requirements not met
- CUDA not available or incompatible
- Training crashes or produces NaN losses
- OOM errors on small models
- Dependency conflicts detected

## Expected Outputs & Logs

### Log Files Generated
```
test_results_YYYYMMDD_HHMMSS/
├── test_imports.log        # Module import test results
├── test_pytorch.log        # PyTorch compatibility results
├── test_cuda.log          # CUDA compatibility results
├── test_transformers.log  # Transformers compatibility results
├── test_training.log      # Minimal training test results
└── summary.txt           # Overall test summary
```

### Success Indicators
1. **Installation**: Package installs without errors
2. **Imports**: All modules load successfully
3. **Compatibility**: Version requirements met
4. **Functionality**: Training loop executes
5. **Memory**: No OOM errors
6. **Performance**: Reasonable throughput

## Troubleshooting Guide

### Common Issues & Solutions

#### Issue 1: CUDA Not Available
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Solution: Install CUDA 12.1+
```

#### Issue 2: Module Import Errors
```bash
# Check installation
pip show zo2

# Solution: Reinstall with dependencies
pip install --upgrade git+https://github.com/liangyuwang/zo2.git
```

#### Issue 3: OOM Errors
```bash
# Check GPU memory
nvidia-smi

# Solution: Reduce batch size or use smaller model
```

#### Issue 4: Version Conflicts
```bash
# Check versions
pip list | grep -E "torch|transformers|accelerate"

# Solution: Update to required versions
pip install torch>=2.4.0 transformers>=4.51.3
```

## Test Commands Summary

### Quick Verification
```bash
python test/installation/quick_test.py
```

### Full Test Suite
```bash
bash test/installation/run_all_tests.sh
```

### Individual Tests
```bash
# Module imports
python test/installation/test_imports.py

# PyTorch compatibility
python test/installation/test_pytorch_version.py

# CUDA compatibility
python test/installation/test_cuda_compatibility.py

# Transformers compatibility
python test/installation/test_transformers_compat.py

# Minimal training
python test/installation/test_minimal_training.py
```

### Memory Testing
```bash
# Record memory for all OPT sizes
bash test/mezo_sgd/hf_opt/record_zo2_memory.sh
```

## Reporting

### Test Report Format
```
Test Name: [Name]
Status: [PASS/FAIL]
Duration: [Time]
Details: [Key findings]
Errors: [If any]
```

### Success Metrics
- 100% of core modules import successfully
- All version requirements met
- Minimal training executes without errors
- Memory usage within expected bounds
- No critical errors or warnings

## Next Steps

After successful Phase 2 testing:
1. Proceed to Phase 3: Memory Testing
2. Test larger models (OPT-350M, OPT-1.3B)
3. Validate multi-GPU functionality
4. Test offloading mechanisms thoroughly
5. Document any issues or limitations found