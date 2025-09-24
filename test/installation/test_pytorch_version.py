#!/usr/bin/env python3
"""
Test PyTorch version compatibility for ZO2.
"""

import sys
import re

def test_pytorch_version():
    """Test PyTorch version and dependencies."""
    print("=" * 60)
    print("PyTorch Version Compatibility Test")
    print("=" * 60)
    
    tests_passed = True
    
    # Test 1: PyTorch import and version
    print("\n1. PyTorch Installation:")
    try:
        import torch
        pytorch_version = torch.__version__
        print(f"   ✓ PyTorch installed: {pytorch_version}")
        
        # Parse version
        version_match = re.match(r'(\d+)\.(\d+)\.(\d+)', pytorch_version)
        if version_match:
            major, minor, patch = map(int, version_match.groups())
            
            # Check if version >= 2.4.0 (required for ZO2)
            if major > 2 or (major == 2 and minor >= 4):
                print(f"   ✓ PyTorch version meets requirements (≥2.4.0)")
            else:
                print(f"   ✗ PyTorch version {pytorch_version} is below requirements (≥2.4.0)")
                tests_passed = False
        else:
            print(f"   ⚠ Could not parse PyTorch version: {pytorch_version}")
    except ImportError as e:
        print(f"   ✗ PyTorch not installed: {e}")
        tests_passed = False
        return 1
    
    # Test 2: Required PyTorch components
    print("\n2. PyTorch Components:")
    
    components = {
        'torch.nn': 'Neural network modules',
        'torch.optim': 'Optimization algorithms',
        'torch.utils.data': 'Data loading utilities',
        'torch.cuda': 'CUDA support',
        'torch.distributed': 'Distributed training',
        'torch.autograd': 'Automatic differentiation',
    }
    
    for module, description in components.items():
        try:
            exec(f"import {module}")
            print(f"   ✓ {module:<20} - {description}")
        except ImportError:
            print(f"   ✗ {module:<20} - {description} NOT AVAILABLE")
            tests_passed = False
    
    # Test 3: Check for mixed precision support
    print("\n3. Mixed Precision Support:")
    try:
        from torch.cuda.amp import autocast, GradScaler
        print("   ✓ Mixed precision training (AMP) available")
    except ImportError:
        print("   ✗ Mixed precision training (AMP) NOT available")
        print("     This may impact training performance")
    
    # Test 4: Check torch compile availability (PyTorch 2.0+)
    print("\n4. Torch Compile Support:")
    if hasattr(torch, 'compile'):
        print(f"   ✓ torch.compile available (PyTorch 2.0+ feature)")
    else:
        print(f"   ⚠ torch.compile not available (requires PyTorch 2.0+)")
    
    # Test 5: Memory management features
    print("\n5. Memory Management Features:")
    
    memory_features = [
        ('empty_cache', 'torch.cuda.empty_cache'),
        ('memory_allocated', 'torch.cuda.memory_allocated'),
        ('max_memory_allocated', 'torch.cuda.max_memory_allocated'),
        ('reset_peak_memory_stats', 'torch.cuda.reset_peak_memory_stats'),
    ]
    
    for feature_name, feature_path in memory_features:
        try:
            feature = eval(feature_path)
            if callable(feature):
                print(f"   ✓ {feature_name:<25} available")
            else:
                print(f"   ✗ {feature_name:<25} NOT callable")
                tests_passed = False
        except (AttributeError, NameError):
            print(f"   ✗ {feature_name:<25} NOT available")
            tests_passed = False
    
    # Test 6: Tensor operations
    print("\n6. Basic Tensor Operations:")
    try:
        # Test basic operations
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        
        # Matrix multiplication
        z = torch.matmul(x, y)
        assert z.shape == (10, 10), "Matrix multiplication failed"
        
        # Gradient computation
        x.requires_grad = True
        loss = (x ** 2).sum()
        loss.backward()
        assert x.grad is not None, "Gradient computation failed"
        
        print("   ✓ Tensor operations working correctly")
    except Exception as e:
        print(f"   ✗ Tensor operations failed: {e}")
        tests_passed = False
    
    # Test 7: Check for important optional dependencies
    print("\n7. Optional Dependencies:")
    
    optional_deps = {
        'torchvision': 'Computer vision utilities',
        'torchaudio': 'Audio processing utilities',
    }
    
    for module, description in optional_deps.items():
        try:
            exec(f"import {module}")
            version = eval(f"{module}.__version__")
            print(f"   ✓ {module:<15} {version:<10} - {description}")
        except ImportError:
            print(f"   ⚠ {module:<15} NOT installed - {description}")
    
    # Summary
    print("\n" + "=" * 60)
    if tests_passed:
        print("✓ All PyTorch compatibility tests passed!")
        return 0
    else:
        print("✗ Some PyTorch compatibility tests failed")
        print("  Please update PyTorch to version 2.4.0 or higher")
        return 1

if __name__ == "__main__":
    sys.exit(test_pytorch_version())