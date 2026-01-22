import torch
import multiprocessing
import platform
import sys

# ============================================
# Define ALL functions at MODULE LEVEL (not inside other functions)
# ============================================

def simple_func(x):
    """Must be defined at module level to be picklable on Windows"""
    return x * 2

def check_system():
    print("=" * 50)
    print("MULTIPROCESSING DIAGNOSTIC")
    print("=" * 50)
    
    # System info
    print(f"\n[SYSTEM]")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # CPU info
    print(f"\n[CPU]")
    print(f"Logical cores: {multiprocessing.cpu_count()}")
    
    # GPU info
    print(f"\n[GPU]")
    if torch.cuda.is_available():
        print(f"CUDA: Available")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"CUDA: Not available")
    
    # Multiprocessing
    print(f"\n[MULTIPROCESSING]")
    methods = multiprocessing.get_all_start_methods()
    print(f"Available methods: {methods}")
    
    # Test if we can spawn processes
    print(f"\n[TEST]")
    try:
        ctx = multiprocessing.get_context('spawn')
        print(" 'spawn' context available")
        
        with ctx.Pool(processes=2) as pool:
            result = pool.map(simple_func, [1, 2, 3])
            print(f" Basic multiprocessing works: {result}")
        
        return True
    except Exception as e:
        print(f" Failed: {e}")
        print("\nTROUBLESHOOTING:")
        print("1. On Windows, ensure all code is under if __name__ == '__main__':")
        print("2. Use 'spawn' method: multiprocessing.set_start_method('spawn')")
        print("3. Avoid lambda functions - use def or classes")
        print("4. Define functions at module level (not inside other functions)")
        return False

if __name__ == "__main__":
    # Set start method for Windows
    multiprocessing.set_start_method('spawn', force=True)
    
    success = check_system()
    print(f"\n{'='*50}")
    print(f"RESULT: {'READY for multiprocessing' if success else 'CHECK REQUIRED'}")
    print("=" * 50)