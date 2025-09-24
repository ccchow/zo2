#!/usr/bin/env python3
"""Test that all ZO2 modules can be imported successfully."""

import sys
import traceback
from typing import List, Tuple

def test_import(module_name: str) -> Tuple[bool, str]:
    """Test importing a module."""
    try:
        __import__(module_name)
        return True, f"✓ {module_name}"
    except ImportError as e:
        return False, f"✗ {module_name}: {str(e)}"
    except Exception as e:
        return False, f"✗ {module_name}: Unexpected error: {str(e)}"

def main():
    """Run all import tests."""
    print("=" * 60)
    print("ZO2 Module Import Tests")
    print("=" * 60)
    
    # Core modules to test
    modules = [
        "zo2",
        "zo2.config",
        "zo2.config.mezo_sgd",
        "zo2.model",
        "zo2.model.huggingface",
        "zo2.model.huggingface.opt",
        "zo2.model.huggingface.qwen3",
        "zo2.model.nanogpt",
        "zo2.optimizer",
        "zo2.optimizer.mezo_sgd",
        "zo2.trainer",
        "zo2.trainer.hf_transformers",
        "zo2.utils",
    ]
    
    # Test specific imports
    specific_imports = [
        ("zo2", ["ZOConfig", "zo_hf_init"]),
        ("zo2.trainer.hf_transformers", ["ZOTrainer", "ZOSFTTrainer"]),
    ]
    
    failed = []
    passed = []
    
    # Test module imports
    print("\n1. Testing module imports:")
    print("-" * 40)
    for module in modules:
        success, msg = test_import(module)
        print(msg)
        if success:
            passed.append(module)
        else:
            failed.append(module)
    
    # Test specific function/class imports
    print("\n2. Testing specific imports:")
    print("-" * 40)
    for module, items in specific_imports:
        try:
            mod = __import__(module, fromlist=items)
            for item in items:
                if hasattr(mod, item):
                    msg = f"✓ from {module} import {item}"
                    print(msg)
                    passed.append(f"{module}.{item}")
                else:
                    msg = f"✗ from {module} import {item}: Not found"
                    print(msg)
                    failed.append(f"{module}.{item}")
        except Exception as e:
            msg = f"✗ from {module} import {items}: {str(e)}"
            print(msg)
            failed.extend([f"{module}.{item}" for item in items])
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print("\nFailed imports:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n✓ All imports successful!")
        sys.exit(0)

if __name__ == "__main__":
    main()