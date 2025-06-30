"""
Hardware detection and optimization for Content Creator
"""
import platform
import subprocess
import sys
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detects hardware capabilities and optimizes settings accordingly"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.hardware_type = self._detect_hardware_type()
        self.optimizations = self._get_optimizations()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "architecture": platform.architecture()[0],
        }
    
    def _detect_hardware_type(self) -> str:
        """Detect the type of hardware available"""
        # Check for Apple Silicon (M1, M2, M3, M4, etc.)
        if self._is_apple_silicon():
            return "apple_silicon"
        
        # Check for NVIDIA GPU
        elif self._has_nvidia_gpu():
            return "nvidia_gpu"
        
        # Check for AMD GPU
        elif self._has_amd_gpu():
            return "amd_gpu"
        
        # Fallback to CPU
        else:
            return "cpu"
    
    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon"""
        if platform.system() != "Darwin":
            return False
        
        machine = platform.machine().lower()
        # Check for Apple Silicon identifiers
        apple_silicon_chips = ["arm64", "aarch64"]
        if any(chip in machine for chip in apple_silicon_chips):
            return True
        
        # Additional check for M-series chips
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                cpu_brand = result.stdout.strip().lower()
                return "apple" in cpu_brand and any(
                    m_chip in cpu_brand for m_chip in ["m1", "m2", "m3", "m4", "m5"]
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return False
    
    def _has_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        # Alternative check using nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _has_amd_gpu(self) -> bool:
        """Check if AMD GPU is available (basic check)"""
        try:
            import torch
            # Check for ROCm support
            return hasattr(torch.version, 'hip') and torch.version.hip is not None
        except ImportError:
            return False
    
    def _get_optimizations(self) -> Dict[str, Any]:
        """Get hardware-specific optimizations"""
        if self.hardware_type == "apple_silicon":
            return self._get_apple_silicon_optimizations()
        elif self.hardware_type == "nvidia_gpu":
            return self._get_nvidia_optimizations()
        elif self.hardware_type == "amd_gpu":
            return self._get_amd_optimizations()
        else:
            return self._get_cpu_optimizations()
    
    def _get_apple_silicon_optimizations(self) -> Dict[str, Any]:
        """Optimizations for Apple Silicon (M1, M2, M3, M4, etc.)"""
        return {
            "device": "mps",
            "precision": "fp16",
            "use_metal": True,
            "memory_efficient": True,
            "max_batch_size": 1,
            "enable_metal_performance_shaders": True,
            "use_unified_memory": True,
            "optimize_for_latency": True,
            "torch_compile": False,  # May not be stable on MPS
            "recommended_models": [
                "AnimateDiff",  # Works well on Apple Silicon
                "Text2Video-Zero",  # Lower memory requirement
                "I2VGen-XL"  # Good balance of quality and performance
            ]
        }
    
    def _get_nvidia_optimizations(self) -> Dict[str, Any]:
        """Optimizations for NVIDIA GPUs"""
        gpu_memory = self._get_gpu_memory()
        
        return {
            "device": "cuda",
            "precision": "fp16",
            "use_xformers": True,
            "memory_efficient": gpu_memory < 12,  # Enable if less than 12GB
            "max_batch_size": min(4, max(1, gpu_memory // 4)),  # Scale with memory
            "enable_flash_attention": True,
            "use_tensorrt": True,
            "torch_compile": True,
            "recommended_models": [
                "VideoCrafter",  # Can utilize high VRAM
                "Stable Video Diffusion",  # Excellent quality
                "LaVie",  # Good for longer videos
                "AnimateDiff"
            ]
        }
    
    def _get_amd_optimizations(self) -> Dict[str, Any]:
        """Optimizations for AMD GPUs"""
        return {
            "device": "cuda",  # ROCm uses CUDA API
            "precision": "fp32",  # More stable on AMD
            "use_xformers": False,
            "memory_efficient": True,
            "max_batch_size": 1,
            "torch_compile": False,
            "recommended_models": [
                "Text2Video-Zero",
                "AnimateDiff"
            ]
        }
    
    def _get_cpu_optimizations(self) -> Dict[str, Any]:
        """Optimizations for CPU-only inference"""
        import multiprocessing
        
        return {
            "device": "cpu",
            "precision": "fp32",
            "use_threading": True,
            "memory_efficient": True,
            "max_batch_size": 1,
            "num_threads": min(8, multiprocessing.cpu_count()),
            "optimize_for_inference": True,
            "torch_compile": False,
            "recommended_models": [
                "Text2Video-Zero",  # Lightest model
                "AnimateDiff"  # Good CPU performance
            ]
        }
    
    def _get_gpu_memory(self) -> int:
        """Get GPU memory in GB"""
        try:
            if self.hardware_type == "nvidia_gpu":
                import torch
                if torch.cuda.is_available():
                    return torch.cuda.get_device_properties(0).total_memory // (1024**3)
            elif self.hardware_type == "apple_silicon":
                # Apple Silicon uses unified memory, estimate based on system RAM
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        total_memory_gb = int(result.stdout.strip()) // (1024**3)
                        # Assume 70% of system memory available for GPU tasks
                        return int(total_memory_gb * 0.7)
                except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                    pass
        except Exception as e:
            logger.warning(f"Could not determine GPU memory: {e}")
        
        return 8  # Default fallback
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        info = {
            "hardware_type": self.hardware_type,
            "system_info": self.system_info,
            "optimizations": self.optimizations,
            "gpu_memory_gb": self._get_gpu_memory(),
        }
        
        # Add device-specific details
        if self.hardware_type == "apple_silicon":
            info["metal_available"] = self._check_metal_availability()
        elif self.hardware_type == "nvidia_gpu":
            info["cuda_version"] = self._get_cuda_version()
            info["gpu_name"] = self._get_gpu_name()
        
        return info
    
    def _check_metal_availability(self) -> bool:
        """Check if Metal Performance Shaders is available"""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            return False
    
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version"""
        try:
            import torch
            return torch.version.cuda
        except ImportError:
            return None
    
    def _get_gpu_name(self) -> Optional[str]:
        """Get GPU name"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        # Alternative method using nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def optimize_for_generation(self, model_name: str, resolution: str, duration: int) -> Dict[str, Any]:
        """Get optimized settings for specific generation parameters"""
        base_opts = self.optimizations.copy()
        
        # Adjust based on resolution
        if resolution in ["4K", "1440p"]:
            base_opts["memory_efficient"] = True
            base_opts["max_batch_size"] = 1
        elif resolution in ["480p", "Square"]:
            base_opts["max_batch_size"] = min(base_opts.get("max_batch_size", 1) * 2, 4)
        
        # Adjust based on duration
        if duration > 30:
            base_opts["memory_efficient"] = True
            base_opts["max_batch_size"] = 1
        elif duration < 10:
            base_opts["max_batch_size"] = min(base_opts.get("max_batch_size", 1) * 2, 4)
        
        # Model-specific adjustments
        model_memory_requirements = {
            "VideoCrafter": 16,
            "Stable Video Diffusion": 12,
            "LaVie": 10,
            "I2VGen-XL": 14,
            "AnimateDiff": 8,
            "Text2Video-Zero": 6,
        }
        
        required_memory = model_memory_requirements.get(model_name, 8)
        available_memory = self._get_gpu_memory()
        
        if available_memory < required_memory:
            base_opts["memory_efficient"] = True
            base_opts["max_batch_size"] = 1
            base_opts["precision"] = "fp16" if self.hardware_type != "cpu" else "fp32"
        
        return base_opts
    
    def get_optimal_device(self) -> str:
        """Get the optimal device for model execution"""
        return self.optimizations.get('device', 'cpu')
    
    def log_hardware_info(self):
        """Log hardware information for debugging"""
        info = self.get_device_info()
        logger.info(f"Hardware Type: {info['hardware_type']}")
        logger.info(f"Platform: {info['system_info']['platform']}")
        logger.info(f"Architecture: {info['system_info']['architecture']}")
        logger.info(f"GPU Memory: {info['gpu_memory_gb']}GB")
        
        if info['hardware_type'] == "apple_silicon":
            logger.info(f"Metal Available: {info.get('metal_available', False)}")
        elif info['hardware_type'] == "nvidia_gpu":
            logger.info(f"CUDA Version: {info.get('cuda_version', 'Unknown')}")
            logger.info(f"GPU Name: {info.get('gpu_name', 'Unknown')}")
        
        logger.info(f"Recommended Models: {info['optimizations'].get('recommended_models', [])}")


# Global hardware detector instance
hardware_detector = HardwareDetector() 