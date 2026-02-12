"""
Description:
Comprehensive safety and monitoring utility for machine learning training on NF's dedicated PC. 
Provides real-time insights into system health, memory usage, GPU status, and overall resource safety. 

Key Features:
1. System Health Logging:
   - Logs RAM, CPU, and GPU usage along with temperatures using `psutil` and `GPUtil`.
   - `log_health(logger)` provides a concise, timestamped status update.

2. Memory and Safety Checks:
   - `memory_is_safe()` evaluates RAM and GPU VRAM against configurable thresholds.
   - `clear_memory_caches()` frees PyTorch GPU cache and triggers Python garbage collection.

3. Detailed System Reports:
   - `get_system_info()` returns a full snapshot of CPU, RAM, GPU, swap, and disk usage.
   - `format_memory_report()` generates a human-readable, safety-aware memory report.

4. GPU and CPU Utilities:
   - Fetch detailed GPU memory and utilization info via PyTorch and GPUtil.
   - Monitor CPU utilization, cores, threads, and frequency.

5. Training-Friendly Utilities:
   - `cooldown(sec)` allows brief pauses to prevent overheating during training loops.
   - Backward-compatible `get_memory_usage_simple()` for legacy scripts.

This script is intended for continuous monitoring during intensive ML training, helping 
prevent out-of-memory errors, overheating, and other system-related interruptions.
"""

import torch
import time
import psutil
import GPUtil
import platform
from datetime import datetime

def log_health(logger):
    """Log current system health status"""
    # RAM usage
    ram_info = psutil.virtual_memory()
    ram_percent = ram_info.percent
    ram_used_gb = ram_info.used / (1024**3)
    ram_total_gb = ram_info.total / (1024**3)
    
    # GPU usage if available
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        gpu_info = f"GPU: {gpu_allocated:.2f}/{gpu_reserved:.2f} GB"
        
        # Try to get GPU utilization from GPUtil
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info += f" | Util: {gpu.load*100:.1f}% | Temp: {gpu.temperature}°C"
        except:
            pass
    else:
        gpu_info = "GPU: Not available"
    
    logger.info(f"RAM: {ram_percent:.1f}% ({ram_used_gb:.1f}/{ram_total_gb:.1f} GB) | {gpu_info}")

def cooldown(sec=0.05):
    """Pause execution briefly to prevent overheating"""
    time.sleep(sec)

def get_memory_usage():
    """Get detailed memory usage information"""
    ram_info = psutil.virtual_memory()
    swap_info = psutil.swap_memory()
    
    memory_data = {
        "ram": {
            "total_gb": ram_info.total / (1024**3),
            "available_gb": ram_info.available / (1024**3),
            "used_gb": ram_info.used / (1024**3),
            "percent": ram_info.percent,
            "free_gb": ram_info.free / (1024**3)
        },
        "swap": {
            "total_gb": swap_info.total / (1024**3),
            "used_gb": swap_info.used / (1024**3),
            "free_gb": swap_info.free / (1024**3),
            "percent": swap_info.percent
        }
    }
    
    # Add CPU info
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_data["cpu"] = {
        "percent": cpu_percent,
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
    }
    
    return memory_data

def get_gpu_memory():
    """Get detailed GPU memory information"""
    gpu_data = {}
    
    if torch.cuda.is_available():
        try:
            # PyTorch GPU info
            gpu_data["pytorch"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0)
            }
            
            # GPUtil info (if available)
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_data["gputil"] = {
                        "name": gpu.name,
                        "total_gb": gpu.memoryTotal / 1024,
                        "used_gb": gpu.memoryUsed / 1024,
                        "free_gb": gpu.memoryFree / 1024,
                        "utilization_percent": gpu.load * 100,
                        "temperature_c": gpu.temperature
                    }
                    
                    # Calculate percentages
                    gpu_data["gputil"]["used_percent"] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                    gpu_data["gputil"]["free_percent"] = (gpu.memoryFree / gpu.memoryTotal) * 100
                    
            except ImportError:
                gpu_data["gputil"] = {"error": "GPUtil not available"}
            
        except Exception as e:
            gpu_data["error"] = str(e)
    else:
        gpu_data["available"] = False
    
    return gpu_data

def get_system_info():
    """Get comprehensive system information"""
    system_info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "python_version": platform.python_version(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add memory info
    system_info["memory"] = get_memory_usage()
    
    # Add GPU info
    system_info["gpu"] = get_gpu_memory()
    
    # Add disk info
    try:
        disk_info = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info[partition.mountpoint] = {
                    "total_gb": usage.total / (1024**3),
                    "used_gb": usage.used / (1024**3),
                    "free_gb": usage.free / (1024**3),
                    "percent": usage.percent,
                    "fstype": partition.fstype
                }
            except:
                pass
        system_info["disk"] = disk_info
    except:
        system_info["disk"] = {"error": "Could not read disk info"}
    
    return system_info

def memory_is_safe(max_ram_percent=85, max_vram_percent=90):
    """Check if memory usage is within safe limits"""
    # Check RAM
    ram_info = psutil.virtual_memory()
    ram_safe = ram_info.percent < max_ram_percent
    
    # Check GPU VRAM
    gpu_safe = True
    gpu_details = {}
    
    if torch.cuda.is_available():
        gpu_mem = get_gpu_memory()
        if "gputil" in gpu_mem and "used_percent" in gpu_mem["gputil"]:
            vram_percent = gpu_mem["gputil"]["used_percent"]
            gpu_safe = vram_percent < max_vram_percent
            gpu_details = {
                "vram_percent": vram_percent,
                "vram_used_gb": gpu_mem["gputil"]["used_gb"],
                "vram_total_gb": gpu_mem["gputil"]["total_gb"],
                "temperature": gpu_mem["gputil"]["temperature_c"]
            }
    
    return {
        "safe": ram_safe and gpu_safe,
        "ram_percent": ram_info.percent,
        "ram_safe": ram_safe,
        "gpu_safe": gpu_safe,
        "gpu_details": gpu_details
    }

def clear_memory_caches():
    """Clear various memory caches to free up memory"""
    # Clear PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Clear Python garbage
    import gc
    gc.collect()
    
    return {
        "cuda_cache_cleared": torch.cuda.is_available(),
        "garbage_collected": True
    }

def format_memory_report():
    """Create a formatted memory report"""
    system_info = get_system_info()
    
    report = "=" * 60 + "\n"
    report += "SYSTEM MEMORY REPORT\n"
    report += "=" * 60 + "\n"
    
    # RAM info
    ram = system_info["memory"]["ram"]
    report += f"RAM: {ram['used_gb']:.1f}/{ram['total_gb']:.1f} GB ({ram['percent']:.1f}%)\n"
    report += f"  Available: {ram['available_gb']:.1f} GB | Free: {ram['free_gb']:.1f} GB\n"
    
    # CPU info
    cpu = system_info["memory"]["cpu"]
    report += f"CPU: {cpu['percent']:.1f}% | Cores: {cpu['cores']} | Threads: {cpu['threads']}\n"
    
    # GPU info
    gpu = system_info["gpu"]
    if gpu.get("available", True) and "gputil" in gpu:
        gpu_info = gpu["gputil"]
        report += f"GPU: {gpu_info.get('name', 'Unknown')}\n"
        report += f"  VRAM: {gpu_info['used_gb']:.1f}/{gpu_info['total_gb']:.1f} GB ({gpu_info['used_percent']:.1f}%)\n"
        report += f"  Utilization: {gpu_info['utilization_percent']:.1f}% | Temp: {gpu_info['temperature_c']}°C\n"
    
    # Safety check
    safety = memory_is_safe()
    report += f"\nSAFETY STATUS: {'SAFE' if safety['safe'] else 'WARNING'}\n"
    if not safety['safe']:
        report += f"  RAM: {safety['ram_percent']:.1f}% {'✓' if safety['ram_safe'] else '✗'}\n"
        if safety['gpu_details']:
            report += f"  VRAM: {safety['gpu_details']['vram_percent']:.1f}% {'✓' if safety['gpu_safe'] else '✗'}\n"
    
    report += "=" * 60
    
    return report

# For backward compatibility
def get_memory_usage_simple():
    """Simple memory usage (compatibility with old code)"""
    ram_info = psutil.virtual_memory()
    gpu_info = get_gpu_memory()
    
    return {
        "ram_percent": ram_info.percent,
        "ram_used_gb": ram_info.used / (1024**3),
        "ram_total_gb": ram_info.total / (1024**3),
        "gpu_info": gpu_info
    }