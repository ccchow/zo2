#!/usr/bin/env python3
"""
Quick installation verification test for ZO2.
Run this immediately after installation to verify basic functionality.
"""

import sys
import os

def quick_test():
    """Quick test to verify ZO2 installation."""
    print("=" * 60)
    print("ZO2 Quick Installation Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Basic imports
    print("\n✓ Testing basic imports...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        import transformers
        print(f"  Transformers version: {transformers.__version__}")
        
        import zo2
        print(f"  ZO2 imported successfully")
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        success = False
    
    # Test 2: CUDA availability
    print("\n✓ Checking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.device_count()} GPU(s)")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print("  ⚠ CUDA not available - CPU mode only")
    except Exception as e:
        print(f"  ✗ CUDA check failed: {e}")
    
    # Test 3: ZO2 core modules
    print("\n✓ Testing ZO2 modules...")
    try:
        from zo2.config.mezo_sgd import MeZOSGDConfig
        print("  ✓ Config module loaded")
        
        from zo2.model.huggingface.opt import mezo_sgd
        print("  ✓ Model module loaded")
        
        from zo2.utils.utils import seed_everything
        print("  ✓ Utils module loaded")
        
    except ImportError as e:
        print(f"  ✗ ZO2 module import failed: {e}")
        success = False
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✓ ZO2 installation verified successfully!")
        print("  You can now run the full test suite with:")
        print("  bash test/installation/run_all_tests.sh")
        return 0
    else:
        print("✗ Installation verification failed")
        print("  Please check the errors above")
        return 1

if __name__ == "__main__":
    sys.exit(quick_test())