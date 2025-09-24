#!/usr/bin/env python3
"""
Test Transformers library compatibility for ZO2.
"""

import sys
import re
from typing import Tuple

def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse version string to tuple of integers."""
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0, 0)

def test_transformers_compatibility():
    """Test Transformers library compatibility."""
    print("=" * 60)
    print("Transformers Library Compatibility Test")
    print("=" * 60)
    
    tests_passed = True
    
    # Test 1: Transformers import and version
    print("\n1. Transformers Installation:")
    try:
        import transformers
        transformers_version = transformers.__version__
        print(f"   ✓ Transformers installed: {transformers_version}")
        
        # Check version requirements (>=4.51.3 for ZO2)
        major, minor, patch = parse_version(transformers_version)
        required = (4, 51, 3)
        
        if (major, minor, patch) >= required:
            print(f"   ✓ Transformers version meets requirements (≥4.51.3)")
        else:
            print(f"   ✗ Transformers version {transformers_version} is below requirements (≥4.51.3)")
            tests_passed = False
    except ImportError as e:
        print(f"   ✗ Transformers not installed: {e}")
        tests_passed = False
        return 1
    
    # Test 2: Required model architectures
    print("\n2. Model Architecture Support:")
    
    models_to_test = [
        ('transformers.models.opt', 'OPT models'),
        ('transformers.models.opt.modeling_opt', 'OPT model implementation'),
        ('transformers.models.opt.configuration_opt', 'OPT configuration'),
        ('transformers.AutoModelForCausalLM', 'Auto model for causal LM'),
        ('transformers.AutoTokenizer', 'Auto tokenizer'),
        ('transformers.Trainer', 'Trainer class'),
        ('transformers.TrainingArguments', 'Training arguments'),
    ]
    
    for module_path, description in models_to_test:
        try:
            parts = module_path.split('.')
            if len(parts) > 1:
                exec(f"from {'.'.join(parts[:-1])} import {parts[-1]}")
            else:
                exec(f"import {module_path}")
            print(f"   ✓ {description:<30} available")
        except ImportError:
            print(f"   ✗ {description:<30} NOT available")
            tests_passed = False
    
    # Test 3: OPT model specific tests
    print("\n3. OPT Model Availability:")
    try:
        from transformers import OPTConfig, OPTModel, OPTForCausalLM
        
        # Test creating a small OPT config
        config = OPTConfig(
            vocab_size=50272,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=2048,
        )
        
        print("   ✓ OPTConfig creation successful")
        print("   ✓ OPTModel available")
        print("   ✓ OPTForCausalLM available")
        
    except Exception as e:
        print(f"   ✗ OPT model components failed: {e}")
        tests_passed = False
    
    # Test 4: Qwen model support (if available)
    print("\n4. Qwen Model Support:")
    try:
        # Qwen models might not be in all transformers versions
        from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
        print("   ✓ Qwen2 models available")
    except ImportError:
        try:
            # Try older Qwen import
            from transformers import QWenConfig, QWenModel
            print("   ✓ Qwen models available (legacy)")
        except ImportError:
            print("   ⚠ Qwen models not available (optional for ZO2)")
    
    # Test 5: Training utilities
    print("\n5. Training Utilities:")
    
    training_utils = [
        'transformers.trainer_utils',
        'transformers.training_args',
        'transformers.optimization',
        'transformers.data.data_collator',
    ]
    
    for module in training_utils:
        try:
            exec(f"import {module}")
            print(f"   ✓ {module.split('.')[-1]:<20} available")
        except ImportError:
            print(f"   ✗ {module.split('.')[-1]:<20} NOT available")
            tests_passed = False
    
    # Test 6: Tokenizer functionality
    print("\n6. Tokenizer Functionality:")
    try:
        from transformers import AutoTokenizer
        
        # Test with a small model name (this won't download if not cached)
        print("   ✓ AutoTokenizer imported successfully")
        
        # Test tokenizer utilities
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        print("   ✓ Tokenizer base classes available")
        
    except ImportError as e:
        print(f"   ✗ Tokenizer functionality failed: {e}")
        tests_passed = False
    
    # Test 7: Dataset handling
    print("\n7. Dataset Utilities:")
    try:
        import datasets
        datasets_version = datasets.__version__
        print(f"   ✓ Datasets library installed: {datasets_version}")
        
        # Check common dataset operations
        from datasets import load_dataset, Dataset, DatasetDict
        print("   ✓ Dataset loading utilities available")
        
    except ImportError:
        print("   ⚠ Datasets library not installed (optional but recommended)")
    
    # Test 8: Accelerate library (used by transformers)
    print("\n8. Accelerate Library:")
    try:
        import accelerate
        accelerate_version = accelerate.__version__
        print(f"   ✓ Accelerate installed: {accelerate_version}")
        
        # Check version (>=1.6.0 recommended)
        major, minor, patch = parse_version(accelerate_version)
        if (major, minor, patch) >= (1, 6, 0):
            print(f"   ✓ Accelerate version meets recommendations (≥1.6.0)")
        else:
            print(f"   ⚠ Accelerate version {accelerate_version} is below recommendations (≥1.6.0)")
            
    except ImportError:
        print("   ⚠ Accelerate not installed (optional but recommended)")
    
    # Test 9: TRL library for RLHF
    print("\n9. TRL Library (for SFT):")
    try:
        import trl
        trl_version = trl.__version__
        print(f"   ✓ TRL installed: {trl_version}")
        
        from trl import SFTTrainer
        print("   ✓ SFTTrainer available")
        
    except ImportError:
        print("   ⚠ TRL not installed (needed for SFT training)")
    
    # Summary
    print("\n" + "=" * 60)
    if tests_passed:
        print("✓ All required Transformers compatibility tests passed!")
        return 0
    else:
        print("✗ Some Transformers compatibility tests failed")
        print("  Please install transformers>=4.51.3")
        return 1

if __name__ == "__main__":
    sys.exit(test_transformers_compatibility())