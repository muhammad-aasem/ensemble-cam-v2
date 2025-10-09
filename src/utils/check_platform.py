#!/usr/bin/env python3
"""
Platform Information Checker for Ensemble CAM Project

This script detects and reports comprehensive system information necessary for
training deep learning models, including:
- Operating System and version
- CPU information (cores, architecture, frequency)
- Memory (RAM) information
- GPU information (NVIDIA, AMD, Intel)
- VRAM (GPU memory) details
- CUDA/ROCm availability
- Python and PyTorch versions
- Storage information
- Network information

The script generates a detailed platform_info.txt file in the project root
with all system specifications and recommendations for training configuration.

Usage:
    uv run python src/utils/check_platform.py

Output:
    - platform_info.txt: Comprehensive system information
    - Console output with key findings and recommendations
"""

import os
import sys
import platform
import subprocess
import json
import psutil
import torch
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import additional GPU detection libraries
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_FILE = PROJECT_ROOT / "platform_info.txt"

class PlatformChecker:
    """Comprehensive platform information checker."""
    
    def __init__(self):
        self.info = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'system': {},
            'cpu': {},
            'memory': {},
            'gpu': {},
            'python': {},
            'pytorch': {},
            'storage': {},
            'network': {},
            'recommendations': {}
        }
    
    def get_system_info(self):
        """Get operating system and basic system information."""
        print("Detecting system information...")
        
        self.info['system'] = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        }
        
        # Additional system info
        try:
            if platform.system() == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
                self.info['system']['windows_version'] = winreg.QueryValueEx(key, "DisplayVersion")[0]
                winreg.CloseKey(key)
        except:
            pass
    
    def get_cpu_info(self):
        """Get CPU information."""
        print("Detecting CPU information...")
        
        self.info['cpu'] = {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'max_frequency': f"{psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() else "Unknown",
            'current_frequency': f"{psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "Unknown",
            'cpu_percent': f"{psutil.cpu_percent(interval=1):.1f}%"
        }
        
        # Try to get more detailed CPU info
        try:
            if platform.system() == "Windows":
                result = subprocess.run(['wmic', 'cpu', 'get', 'name,numberofcores,numberoflogicalprocessors,maxclockspeed', '/format:csv'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        parts = lines[1].split(',')
                        if len(parts) >= 4:
                            self.info['cpu']['name'] = parts[1].strip()
                            self.info['cpu']['physical_cores_wmic'] = parts[2].strip()
                            self.info['cpu']['logical_cores_wmic'] = parts[3].strip()
                            self.info['cpu']['max_clock_speed'] = f"{parts[4].strip()} MHz"
        except:
            pass
    
    def get_memory_info(self):
        """Get memory (RAM) information."""
        print("Detecting memory information...")
        
        memory = psutil.virtual_memory()
        self.info['memory'] = {
            'total_gb': f"{memory.total / (1024**3):.2f} GB",
            'available_gb': f"{memory.available / (1024**3):.2f} GB",
            'used_gb': f"{memory.used / (1024**3):.2f} GB",
            'used_percent': f"{memory.percent:.1f}%",
            'total_bytes': memory.total,
            'available_bytes': memory.available
        }
    
    def get_gpu_info(self):
        """Get GPU information including VRAM."""
        print("Detecting GPU information...")
        
        gpu_info = {
            'nvidia_gpus': [],
            'amd_gpus': [],
            'intel_gpus': [],
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
        
        # Check NVIDIA GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total_gb': f"{torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB",
                    'memory_total_bytes': torch.cuda.get_device_properties(i).total_memory,
                    'compute_capability': f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}",
                    'multiprocessor_count': torch.cuda.get_device_properties(i).multi_processor_count
                }
                gpu_info['nvidia_gpus'].append(gpu)
        
        # Try to get additional GPU info using system commands
        try:
            if platform.system() == "Windows":
                # Get all video controllers
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name,AdapterRAM,DriverVersion', '/format:csv'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        if line.strip():
                            parts = line.split(',')
                            if len(parts) >= 4:
                                name = parts[1].strip()
                                memory = parts[2].strip()
                                driver = parts[3].strip()
                                
                                if 'nvidia' in name.lower():
                                    gpu_info['nvidia_gpus'].append({
                                        'name': name,
                                        'memory_bytes': int(memory) if memory.isdigit() else 0,
                                        'memory_gb': f"{int(memory) / (1024**3):.2f} GB" if memory.isdigit() else "Unknown",
                                        'driver_version': driver
                                    })
                                elif 'amd' in name.lower() or 'radeon' in name.lower():
                                    gpu_info['amd_gpus'].append({
                                        'name': name,
                                        'memory_bytes': int(memory) if memory.isdigit() else 0,
                                        'memory_gb': f"{int(memory) / (1024**3):.2f} GB" if memory.isdigit() else "Unknown",
                                        'driver_version': driver
                                    })
                                elif 'intel' in name.lower():
                                    gpu_info['intel_gpus'].append({
                                        'name': name,
                                        'memory_bytes': int(memory) if memory.isdigit() else 0,
                                        'memory_gb': f"{int(memory) / (1024**3):.2f} GB" if memory.isdigit() else "Unknown",
                                        'driver_version': driver
                                    })
        except:
            pass
        
        # Try nvidia-smi for additional NVIDIA info
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,cuda_version', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        gpu_info['nvidia_gpus'].append({
                            'name': parts[0],
                            'memory_total_mb': parts[1],
                            'driver_version': parts[2],
                            'cuda_version': parts[3]
                        })
        except:
            pass
        
        self.info['gpu'] = gpu_info
    
    def get_python_info(self):
        """Get Python environment information."""
        print("Detecting Python environment...")
        
        self.info['python'] = {
            'version': sys.version,
            'executable': sys.executable,
            'platform': sys.platform,
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler()
        }
        
        # Get installed packages
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                packages = {}
                lines = result.stdout.strip().split('\n')[2:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            packages[parts[0]] = parts[1]
                self.info['python']['installed_packages'] = packages
        except:
            pass
    
    def get_pytorch_info(self):
        """Get PyTorch information."""
        print("Detecting PyTorch information...")
        
        self.info['pytorch'] = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        # Get current device
        if torch.cuda.is_available():
            self.info['pytorch']['current_device'] = torch.cuda.current_device()
            self.info['pytorch']['current_device_name'] = torch.cuda.get_device_name()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.info['pytorch']['current_device'] = 'mps'
        else:
            self.info['pytorch']['current_device'] = 'cpu'
    
    def get_storage_info(self):
        """Get storage information."""
        print("Detecting storage information...")
        
        storage_info = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                storage_info[partition.mountpoint] = {
                    'device': partition.device,
                    'fstype': partition.fstype,
                    'total_gb': f"{usage.total / (1024**3):.2f} GB",
                    'used_gb': f"{usage.used / (1024**3):.2f} GB",
                    'free_gb': f"{usage.free / (1024**3):.2f} GB",
                    'used_percent': f"{(usage.used / usage.total) * 100:.1f}%"
                }
            except:
                pass
        
        self.info['storage'] = storage_info
    
    def get_network_info(self):
        """Get network information."""
        print("Detecting network information...")
        
        try:
            # Get network interfaces
            interfaces = psutil.net_if_addrs()
            network_info = {}
            
            for interface, addresses in interfaces.items():
                network_info[interface] = []
                for addr in addresses:
                    network_info[interface].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask,
                        'broadcast': addr.broadcast
                    })
            
            self.info['network'] = network_info
        except:
            pass
    
    def generate_recommendations(self):
        """Generate training recommendations based on system specs."""
        print("Generating training recommendations...")
        
        recommendations = {
            'gpu_recommendation': 'Unknown',
            'batch_size_recommendation': 'Unknown',
            'model_recommendation': 'Unknown',
            'training_time_estimate': 'Unknown',
            'memory_warnings': [],
            'optimization_suggestions': []
        }
        
        # GPU recommendations
        if self.info['gpu']['nvidia_gpus']:
            best_gpu = max(self.info['gpu']['nvidia_gpus'], 
                          key=lambda x: x.get('memory_total_bytes', 0))
            vram_gb = best_gpu.get('memory_total_bytes', 0) / (1024**3)
            
            if vram_gb >= 24:
                recommendations['gpu_recommendation'] = 'Excellent - High-end GPU with plenty of VRAM'
                recommendations['batch_size_recommendation'] = '32-64 (can handle large models)'
                recommendations['model_recommendation'] = 'All models including ViT and large EfficientNet'
            elif vram_gb >= 12:
                recommendations['gpu_recommendation'] = 'Very Good - Mid-high end GPU'
                recommendations['batch_size_recommendation'] = '16-32 (good for most models)'
                recommendations['model_recommendation'] = 'Most models except very large ones'
            elif vram_gb >= 8:
                recommendations['gpu_recommendation'] = 'Good - Mid-range GPU'
                recommendations['batch_size_recommendation'] = '8-16 (moderate batch sizes)'
                recommendations['model_recommendation'] = 'ResNet, DenseNet, MobileNet, EfficientNet-B4'
            elif vram_gb >= 4:
                recommendations['gpu_recommendation'] = 'Fair - Entry-level GPU'
                recommendations['batch_size_recommendation'] = '4-8 (small batch sizes)'
                recommendations['model_recommendation'] = 'ResNet, DenseNet, MobileNet (smaller models)'
            else:
                recommendations['gpu_recommendation'] = 'Limited - Low VRAM GPU'
                recommendations['batch_size_recommendation'] = '2-4 (very small batches)'
                recommendations['model_recommendation'] = 'MobileNet, small ResNet variants only'
                recommendations['memory_warnings'].append('Very limited VRAM - consider CPU training')
        elif self.info['gpu']['amd_gpus']:
            recommendations['gpu_recommendation'] = 'AMD GPU detected - may work with ROCm'
            recommendations['batch_size_recommendation'] = 'Start with 8-16 and adjust'
            recommendations['model_recommendation'] = 'Test with ResNet first'
            recommendations['optimization_suggestions'].append('Consider installing ROCm for AMD GPU support')
        elif self.info['gpu']['intel_gpus']:
            recommendations['gpu_recommendation'] = 'Intel integrated graphics - not suitable for training'
            recommendations['batch_size_recommendation'] = 'N/A - will use CPU'
            recommendations['model_recommendation'] = 'CPU training only - very slow'
            recommendations['memory_warnings'].append('No dedicated GPU - training will be very slow on CPU')
        else:
            recommendations['gpu_recommendation'] = 'No GPU detected - CPU training only'
            recommendations['batch_size_recommendation'] = 'N/A - will use CPU'
            recommendations['model_recommendation'] = 'CPU training only - very slow'
            recommendations['memory_warnings'].append('No GPU detected - training will be very slow on CPU')
        
        # RAM recommendations
        ram_gb = self.info['memory']['total_bytes'] / (1024**3)
        if ram_gb < 8:
            recommendations['memory_warnings'].append('Low RAM (<8GB) - may cause memory issues')
        elif ram_gb < 16:
            recommendations['optimization_suggestions'].append('Consider 16GB+ RAM for better performance')
        
        # Training time estimates
        if self.info['gpu']['nvidia_gpus']:
            vram_gb = max([gpu.get('memory_total_bytes', 0) for gpu in self.info['gpu']['nvidia_gpus']]) / (1024**3)
            if vram_gb >= 12:
                recommendations['training_time_estimate'] = 'Fast (1-3 hours for 50 epochs)'
            elif vram_gb >= 6:
                recommendations['training_time_estimate'] = 'Moderate (3-6 hours for 50 epochs)'
            else:
                recommendations['training_time_estimate'] = 'Slow (6-12 hours for 50 epochs)'
        else:
            recommendations['training_time_estimate'] = 'Very Slow (12-24+ hours for 50 epochs on CPU)'
        
        # Storage recommendations
        for mount, info in self.info['storage'].items():
            if 'used_percent' in info:
                used_percent = float(info['used_percent'].replace('%', ''))
                if used_percent > 90:
                    recommendations['memory_warnings'].append(f'Storage {mount} is {used_percent:.1f}% full')
        
        self.info['recommendations'] = recommendations
    
    def save_to_file(self):
        """Save platform information to file."""
        print(f"Saving platform information to {OUTPUT_FILE}...")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("ENSEMBLE CAM - PLATFORM INFORMATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {self.info['timestamp']}\n")
            f.write(f"Project: Ensemble CAM - NIH Chest X-ray Classification\n")
            f.write("=" * 60 + "\n\n")
            
            # System Information
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 20 + "\n")
            for key, value in self.info['system'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # CPU Information
            f.write("CPU INFORMATION\n")
            f.write("-" * 20 + "\n")
            for key, value in self.info['cpu'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # Memory Information
            f.write("MEMORY INFORMATION\n")
            f.write("-" * 20 + "\n")
            for key, value in self.info['memory'].items():
                if 'bytes' not in key:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # GPU Information
            f.write("GPU INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"CUDA Available: {self.info['gpu']['cuda_available']}\n")
            f.write(f"CUDA Version: {self.info['gpu']['cuda_version']}\n")
            f.write(f"CUDA Device Count: {self.info['gpu']['cuda_device_count']}\n")
            f.write(f"MPS Available: {self.info['gpu']['mps_available']}\n")
            f.write("\n")
            
            if self.info['gpu']['nvidia_gpus']:
                f.write("NVIDIA GPUs:\n")
                for i, gpu in enumerate(self.info['gpu']['nvidia_gpus']):
                    f.write(f"  GPU {i}:\n")
                    for key, value in gpu.items():
                        f.write(f"    {key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
            
            if self.info['gpu']['amd_gpus']:
                f.write("AMD GPUs:\n")
                for i, gpu in enumerate(self.info['gpu']['amd_gpus']):
                    f.write(f"  GPU {i}:\n")
                    for key, value in gpu.items():
                        f.write(f"    {key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
            
            if self.info['gpu']['intel_gpus']:
                f.write("Intel GPUs:\n")
                for i, gpu in enumerate(self.info['gpu']['intel_gpus']):
                    f.write(f"  GPU {i}:\n")
                    for key, value in gpu.items():
                        f.write(f"    {key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
            
            # PyTorch Information
            f.write("PYTORCH INFORMATION\n")
            f.write("-" * 20 + "\n")
            for key, value in self.info['pytorch'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # Storage Information
            f.write("STORAGE INFORMATION\n")
            f.write("-" * 20 + "\n")
            for mount, info in self.info['storage'].items():
                f.write(f"Mount: {mount}\n")
                for key, value in info.items():
                    f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")
            
            # Recommendations
            f.write("TRAINING RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for key, value in self.info['recommendations'].items():
                if isinstance(value, list):
                    f.write(f"{key.replace('_', ' ').title()}:\n")
                    for item in value:
                        f.write(f"  - {item}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # JSON dump for programmatic access
            f.write("RAW DATA (JSON)\n")
            f.write("-" * 20 + "\n")
            f.write(json.dumps(self.info, indent=2, default=str))
    
    def print_summary(self):
        """Print a summary of key findings."""
        print("\n" + "=" * 60)
        print("PLATFORM CHECK SUMMARY")
        print("=" * 60)
        
        print(f"System: {self.info['system']['system']} {self.info['system']['release']}")
        print(f"CPU: {self.info['cpu']['logical_cores']} logical cores")
        print(f"RAM: {self.info['memory']['total_gb']}")
        
        if self.info['gpu']['nvidia_gpus']:
            best_gpu = max(self.info['gpu']['nvidia_gpus'], 
                          key=lambda x: x.get('memory_total_bytes', 0))
            print(f"GPU: {best_gpu.get('name', 'Unknown')} ({best_gpu.get('memory_total_gb', 'Unknown')})")
            print(f"CUDA: {self.info['gpu']['cuda_version']}")
        elif self.info['gpu']['amd_gpus']:
            print(f"GPU: AMD GPU detected")
        elif self.info['gpu']['intel_gpus']:
            print(f"GPU: Intel integrated graphics")
        else:
            print(f"GPU: No dedicated GPU detected")
        
        print(f"PyTorch: {self.info['pytorch']['version']}")
        print(f"Current Device: {self.info['pytorch']['current_device']}")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 20)
        print(f"GPU Status: {self.info['recommendations']['gpu_recommendation']}")
        print(f"Batch Size: {self.info['recommendations']['batch_size_recommendation']}")
        print(f"Models: {self.info['recommendations']['model_recommendation']}")
        print(f"Training Time: {self.info['recommendations']['training_time_estimate']}")
        
        if self.info['recommendations']['memory_warnings']:
            print("\nWARNINGS:")
            for warning in self.info['recommendations']['memory_warnings']:
                print(f"  ⚠️  {warning}")
        
        if self.info['recommendations']['optimization_suggestions']:
            print("\nSUGGESTIONS:")
            for suggestion in self.info['recommendations']['optimization_suggestions']:
                print(f"  💡 {suggestion}")
        
        print(f"\nDetailed report saved to: {OUTPUT_FILE}")
    
    def run(self):
        """Run the complete platform check."""
        print("Ensemble CAM - Platform Information Checker")
        print("=" * 50)
        
        try:
            self.get_system_info()
            self.get_cpu_info()
            self.get_memory_info()
            self.get_gpu_info()
            self.get_python_info()
            self.get_pytorch_info()
            self.get_storage_info()
            self.get_network_info()
            self.generate_recommendations()
            self.save_to_file()
            self.print_summary()
            
            return True
            
        except Exception as e:
            print(f"Error during platform check: {e}")
            return False

def main():
    """Main function."""
    checker = PlatformChecker()
    success = checker.run()
    
    if success:
        print("\n✅ Platform check completed successfully!")
        return 0
    else:
        print("\n❌ Platform check failed!")
        return 1

if __name__ == "__main__":
    exit(main())
