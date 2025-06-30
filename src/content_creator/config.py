"""
Configuration settings for Content Creator
"""
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class Config:
    """Main configuration class for Content Creator"""
    
    # Video Generation Models - List of available models
    SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "Stable Video Diffusion": {
            "model_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "type": "diffusion",
            "description": "High-quality video generation from images",
            "memory_requirement": "12GB",
            "supports_text_prompt": False,
            "supports_image_input": True,
            "max_duration": 25,  # frames
        },
        "AnimateDiff": {
            "model_id": "guoyww/animatediff-motion-adapter-v1-5-2",
            "type": "diffusion",
            "description": "Animation-focused video generation",
            "memory_requirement": "8GB",
            "supports_text_prompt": True,
            "supports_image_input": True,
            "max_duration": 60,  # seconds
        },
        "Text2Video-Zero": {
            "model_id": "damo-vilab/text-to-video-ms-1.7b",
            "type": "transformer",
            "description": "Text-to-video generation",
            "memory_requirement": "6GB",
            "supports_text_prompt": True,
            "supports_image_input": False,
            "max_duration": 30,  # seconds
        },
        "VideoCrafter": {
            "model_id": "VideoCrafter/VideoCrafter2",
            "type": "diffusion",
            "description": "High-resolution video synthesis",
            "memory_requirement": "16GB",
            "supports_text_prompt": True,
            "supports_image_input": True,
            "max_duration": 120,  # seconds
        },
        "LaVie": {
            "model_id": "vchitect/LaVie",
            "type": "transformer",
            "description": "Long video generation",
            "memory_requirement": "10GB",
            "supports_text_prompt": True,
            "supports_image_input": True,
            "max_duration": 180,  # seconds
        },
        "I2VGen-XL": {
            "model_id": "ali-vilab/i2vgen-xl",
            "type": "diffusion",
            "description": "Image-to-video generation",
            "memory_requirement": "14GB",
            "supports_text_prompt": True,
            "supports_image_input": True,
            "max_duration": 45,  # seconds
        }
    })
    
    # Default model selection
    DEFAULT_MODEL: str = "AnimateDiff"
    
    # Image Generation Models - List of available models for image generation
    SUPPORTED_IMAGE_MODELS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "Flux-NSFW-uncensored": {
            "model_id": "Heartsync/Flux-NSFW-uncensored",
            "base_model": "black-forest-labs/FLUX.1-dev",
            "type": "flux_lora",
            "description": "Uncensored image generation with FLUX.1-dev + LoRA",
            "memory_requirement": "12GB",
            "supports_text_prompt": True,
            "supports_image_input": False,
            "max_resolution": "2048x2048",
            "lora_weight": "lora.safetensors",
            "not_for_all_audiences": True,
        },
        "FLUX.1-dev": {
            "model_id": "black-forest-labs/FLUX.1-dev",
            "type": "flux",
            "description": "High-quality text-to-image generation with FLUX.1-dev",
            "memory_requirement": "12GB",
            "supports_text_prompt": True,
            "supports_image_input": False,
            "max_resolution": "2048x2048",
        },
        "FLUX.1-schnell": {
            "model_id": "black-forest-labs/FLUX.1-schnell",
            "type": "flux",
            "description": "Fast text-to-image generation with FLUX.1-schnell",
            "memory_requirement": "8GB",
            "supports_text_prompt": True,
            "supports_image_input": False,
            "max_resolution": "1024x1024",
        },
        "Stable Diffusion XL": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "type": "diffusion",
            "description": "High-resolution image generation with SDXL",
            "memory_requirement": "8GB",
            "supports_text_prompt": True,
            "supports_image_input": False,
            "max_resolution": "1024x1024",
        },
        "Stable Diffusion 2.1": {
            "model_id": "stabilityai/stable-diffusion-2-1",
            "type": "diffusion",
            "description": "Popular text-to-image generation model",
            "memory_requirement": "6GB",
            "supports_text_prompt": True,
            "supports_image_input": False,
            "max_resolution": "768x768",
        },
        "Midjourney Style": {
            "model_id": "prompthero/openjourney-v4",
            "type": "diffusion",
            "description": "Midjourney-style image generation",
            "memory_requirement": "6GB",
            "supports_text_prompt": True,
            "supports_image_input": False,
            "max_resolution": "768x768",
        },
        "DreamShaper": {
            "model_id": "Lykon/DreamShaper",
            "type": "diffusion",
            "description": "Versatile image generation model",
            "memory_requirement": "6GB",
            "supports_text_prompt": True,
            "supports_image_input": False,
            "max_resolution": "768x768",
        },
        "Realistic Vision": {
            "model_id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
            "type": "diffusion",
            "description": "Photorealistic image generation",
            "memory_requirement": "6GB",
            "supports_text_prompt": True,
            "supports_image_input": False,
            "max_resolution": "768x768",
        }
    })
    
    # Default image model
    DEFAULT_IMAGE_MODEL: str = "Stable Diffusion XL"
    
    # Video Styles
    VIDEO_STYLES: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "Realistic": {
            "description": "Photorealistic video generation",
            "style_prompt_suffix": ", photorealistic, cinematic, high quality, detailed",
            "negative_prompt": "cartoon, anime, painting, sketch, low quality, blurry",
            "guidance_scale": 7.5,
        },
        "Anime": {
            "description": "Animated/cartoon style",
            "style_prompt_suffix": ", anime style, animated, cartoon, vibrant colors",
            "negative_prompt": "photorealistic, real photo, low quality, blurry",
            "guidance_scale": 9.0,
        },
        "Cinematic": {
            "description": "Movie-like cinematic style",
            "style_prompt_suffix": ", cinematic, film grain, dramatic lighting, professional",
            "negative_prompt": "amateur, low quality, blurry, oversaturated",
            "guidance_scale": 8.0,
        },
        "Artistic": {
            "description": "Artistic and stylized",
            "style_prompt_suffix": ", artistic, stylized, creative, unique perspective",
            "negative_prompt": "boring, plain, low quality, blurry",
            "guidance_scale": 8.5,
        },
        "Documentary": {
            "description": "Documentary-style footage",
            "style_prompt_suffix": ", documentary style, natural lighting, realistic",
            "negative_prompt": "fantasy, surreal, low quality, blurry",
            "guidance_scale": 6.5,
        },
        "Fantasy": {
            "description": "Fantasy and magical themes",
            "style_prompt_suffix": ", fantasy, magical, ethereal, mystical",
            "negative_prompt": "mundane, ordinary, low quality, blurry",
            "guidance_scale": 9.5,
        },
        "Custom": {
            "description": "Style guided by reference images",
            "style_prompt_suffix": "",
            "negative_prompt": "low quality, blurry",
            "guidance_scale": 7.0,
        }
    })
    
    # Default style
    DEFAULT_STYLE: str = "Realistic"
    
    # Supported Resolutions (width, height) - All dimensions divisible by 8 for AI models
    RESOLUTIONS: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "480p": (856, 480),  # Fixed: 856 is divisible by 8 (was 854)
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4K": (3840, 2160),
        "Square": (1024, 1024),
        "Portrait": (720, 1280),
        "Widescreen": (2560, 1080),
    })
    
    # Default resolution
    DEFAULT_RESOLUTION: str = "720p"
    
    # Video Output Formats
    OUTPUT_FORMATS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "MP4": {
            "extension": ".mp4",
            "codec": "libx264",
            "description": "Most compatible format",
            "quality": "high",
        },
        "WebM": {
            "extension": ".webm",
            "codec": "libvpx-vp9",
            "description": "Web-optimized format",
            "quality": "high",
        },
        "AVI": {
            "extension": ".avi",
            "codec": "libxvid",
            "description": "Legacy format with good compatibility",
            "quality": "medium",
        },
        "MOV": {
            "extension": ".mov",
            "codec": "libx264",
            "description": "QuickTime format (good for Mac)",
            "quality": "high",
        },
        "GIF": {
            "extension": ".gif",
            "codec": "gif",
            "description": "Animated GIF (lower quality, smaller size)",
            "quality": "low",
        }
    })
    
    # Default output format
    DEFAULT_OUTPUT_FORMAT: str = "MP4"
    
    # Duration settings (in seconds)
    MIN_DURATION: int = 1
    MAX_DURATION: int = 180
    DEFAULT_DURATION: int = 10
    
    # Frame rate options
    FRAME_RATES: List[int] = field(default_factory=lambda: [15, 24, 30, 60])
    DEFAULT_FRAME_RATE: int = 24
    
    # Image input settings
    SUPPORTED_IMAGE_FORMATS: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"
    ])
    MAX_REFERENCE_IMAGES: int = 5
    MAX_IMAGE_SIZE: Tuple[int, int] = (2048, 2048)
    
    # Quality settings
    QUALITY_PRESETS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "Draft": {
            "description": "Fast generation, lower quality",
            "inference_steps": 10,
            "guidance_scale": 5.0,
            "resolution_factor": 0.5,
        },
        "Balanced": {
            "description": "Good balance of speed and quality",
            "inference_steps": 20,
            "guidance_scale": 7.5,
            "resolution_factor": 1.0,
        },
        "High": {
            "description": "High quality, slower generation",
            "inference_steps": 50,
            "guidance_scale": 9.0,
            "resolution_factor": 1.0,
        },
        "Ultra": {
            "description": "Maximum quality, very slow",
            "inference_steps": 100,
            "guidance_scale": 12.0,
            "resolution_factor": 1.2,
        }
    })
    
    # Default quality
    DEFAULT_QUALITY: str = "Balanced"
    
    # Hardware-specific settings
    HARDWARE_CONFIGS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "apple_silicon": {
            "device": "mps",
            "use_metal": True,
            "memory_efficient": True,
            "max_batch_size": 1,
            "precision": "fp16",
        },
        "nvidia_gpu": {
            "device": "cuda",
            "use_xformers": True,
            "memory_efficient": False,
            "max_batch_size": 4,
            "precision": "fp16",
        },
        "cpu": {
            "device": "cpu",
            "use_threading": True,
            "memory_efficient": True,
            "max_batch_size": 1,
            "precision": "fp32",
        }
    })
    
    # File and directory settings
    OUTPUT_DIR: str = "outputs"
    TEMP_DIR: str = "temp"
    MODEL_CACHE_DIR: str = "models"  # Local models directory within repo
    
    # UI Settings
    GRADIO_THEME: str = "soft"
    GRADIO_PORT: int = 7860
    GRADIO_HOST: str = "127.0.0.1"
    GRADIO_SHARE: bool = False
    
    # Performance settings
    MAX_CONCURRENT_GENERATIONS: int = 1
    CLEANUP_TEMP_FILES: bool = True
    AUTO_OPTIMIZE_SETTINGS: bool = True
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        os.makedirs(self.MODEL_CACHE_DIR, exist_ok=True)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        return self.SUPPORTED_MODELS.get(model_name, {})
    
    def get_style_info(self, style_name: str) -> Dict[str, Any]:
        """Get information about a specific style"""
        return self.VIDEO_STYLES.get(style_name, {})
    
    def get_resolution(self, resolution_name: str) -> Tuple[int, int]:
        """Get resolution dimensions"""
        return self.RESOLUTIONS.get(resolution_name, self.RESOLUTIONS[self.DEFAULT_RESOLUTION])
    
    def get_quality_settings(self, quality_name: str) -> Dict[str, Any]:
        """Get quality preset settings"""
        return self.QUALITY_PRESETS.get(quality_name, self.QUALITY_PRESETS[self.DEFAULT_QUALITY])
    
    def get_output_format_info(self, format_name: str) -> Dict[str, Any]:
        """Get output format information"""
        return self.OUTPUT_FORMATS.get(format_name, self.OUTPUT_FORMATS[self.DEFAULT_OUTPUT_FORMAT])
    
    def validate_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Ensure resolution dimensions are divisible by 8 for AI models"""
        def round_to_8(value: int) -> int:
            return ((value + 7) // 8) * 8
        
        return (round_to_8(width), round_to_8(height))
    
    def get_validated_resolution(self, resolution_name: str) -> Tuple[int, int]:
        """Get resolution with validation that dimensions are divisible by 8"""
        width, height = self.get_resolution(resolution_name)
        return self.validate_resolution(width, height)


# Global config instance
config = Config() 