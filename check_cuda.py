"""
CUDA Check Script
Verifies CUDA installation and GPU availability
"""
import sys

try:
    import torch
    print("=" * 60)
    print("CUDA & GPU CHECK")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version.split()[0]}")
    print()
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print()
        
        # Print GPU details
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print()
        
        # Test GPU computation
        print("Testing GPU computation...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("[OK] GPU computation test passed!")
        print()
        
        print("=" * 60)
        print("[OK] CUDA is properly configured and working!")
        print("=" * 60)
        sys.exit(0)
    else:
        print()
        print("=" * 60)
        print("[ERROR] CUDA is not available!")
        print("=" * 60)
        print("Possible reasons:")
        print("1. No NVIDIA GPU detected")
        print("2. CUDA drivers not installed")
        print("3. PyTorch was installed without CUDA support")
        print()
        print("To install PyTorch with CUDA support, visit:")
        print("https://pytorch.org/get-started/locally/")
        print()
        print("Or run:")
        print("  pip uninstall torch torchvision")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("=" * 60)
        sys.exit(1)
        
except ImportError:
    print("=" * 60)
    print("[ERROR] PyTorch is not installed!")
    print("=" * 60)
    print("Please install PyTorch first:")
    print("  pip install torch torchvision")
    print("=" * 60)
    sys.exit(1)
except Exception as e:
    print("=" * 60)
    print(f"[ERROR] Error: {e}")
    print("=" * 60)
    sys.exit(1)

