"""
Configuration settings for Content Creator
"""
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed


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
    
    # Default model selection (configurable via environment)
    DEFAULT_MODEL: str = os.getenv("DEFAULT_VIDEO_MODEL", "AnimateDiff")
    
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
    
    # Default image model (configurable via environment)
    DEFAULT_IMAGE_MODEL: str = os.getenv("DEFAULT_IMAGE_MODEL", "Stable Diffusion XL")
    
    # Video Styles
    VIDEO_STYLES: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "None": {
            "description": "No specific style applied",
            "style_prompt_suffix": "",
            "negative_prompt": "low quality, blurry",
            "guidance_scale": 7.5,
        },
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
        "Hentai": {
            "description": "Hentai/adult anime style",
            "style_prompt_suffix": ", hentai style, anime, detailed, high quality, nsfw",
            "negative_prompt": "photorealistic, realistic, censored, low quality, blurry, ugly",
            "guidance_scale": 9.5,
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
        "Reference": {
            "description": "Style guided by reference images",
            "style_prompt_suffix": "",
            "negative_prompt": "low quality, blurry",
            "guidance_scale": 7.0,
        }
    })
    
    # Default style
    DEFAULT_STYLE: str = "Realistic"
    
    # Supported Resolutions (width, height) - Optimized for different AI models
    RESOLUTIONS: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "480p": (864, 480),  # Fixed: 864 is divisible by 16 for FLUX compatibility
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4K": (3840, 2160),
        "Square": (1024, 1024),
        "Portrait": (720, 1280),
        "Widescreen": (2560, 1080),
    })
    
    # Default resolution (configurable via environment)
    DEFAULT_RESOLUTION: str = os.getenv("DEFAULT_RESOLUTION", "720p")
    
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
    
    # Default quality (configurable via environment)
    DEFAULT_QUALITY: str = os.getenv("DEFAULT_QUALITY", "Balanced")
    
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
    
    # File and directory settings (configurable via environment)
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    MODEL_CACHE_DIR: str = os.getenv("CACHE_DIR", "models")  # Local models directory within repo
    
    # UI Settings (configurable via environment variables)
    GRADIO_THEME: str = os.getenv("GRADIO_THEME", "soft")
    GRADIO_PORT: int = int(os.getenv("PORT", "80" if os.getenv("ENVIRONMENT") == "production" else "80"))
    GRADIO_HOST: str = os.getenv("HOST", "0.0.0.0" if os.getenv("ENVIRONMENT") == "production" else "127.0.0.1")
    GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    
    # Performance settings (configurable via environment)
    MAX_CONCURRENT_GENERATIONS: int = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "1"))
    CLEANUP_TEMP_FILES: bool = os.getenv("CLEANUP_TEMP_FILES", "true").lower() == "true"
    AUTO_OPTIMIZE_SETTINGS: bool = os.getenv("AUTO_OPTIMIZE_SETTINGS", "true").lower() == "true"
    
    # Image Styles (specific for image generation)
    IMAGE_STYLES: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "None": {
            "description": "No specific style applied",
            "style_prompt_suffix": "",
            "negative_prompt": "low quality, blurry",
            "guidance_scale": 7.5,
        },
        "Realistic": {
            "description": "Photorealistic image generation",
            "style_prompt_suffix": ", photorealistic, detailed, high quality",
            "negative_prompt": "cartoon, anime, painting, sketch, low quality, blurry",
            "guidance_scale": 7.5,
        },
        "Anime": {
            "description": "Anime/manga style",
            "style_prompt_suffix": ", anime style, manga, vibrant colors, detailed",
            "negative_prompt": "photorealistic, real photo, low quality, blurry",
            "guidance_scale": 9.0,
        },
        "Hentai": {
            "description": "Hentai/adult anime style",
            "style_prompt_suffix": ", hentai style, anime, detailed, high quality, nsfw, uncensored",
            "negative_prompt": "photorealistic, realistic, censored, low quality, blurry, ugly, bad anatomy",
            "guidance_scale": 9.5,
        },
        "Artistic": {
            "description": "Artistic and stylized",
            "style_prompt_suffix": ", artistic, stylized, creative, masterpiece",
            "negative_prompt": "boring, plain, low quality, blurry",
            "guidance_scale": 8.5,
        },
        "Fantasy": {
            "description": "Fantasy and magical themes",
            "style_prompt_suffix": ", fantasy, magical, ethereal, mystical, detailed",
            "negative_prompt": "mundane, ordinary, low quality, blurry",
            "guidance_scale": 9.0,
        },
        "Digital Art": {
            "description": "Digital artwork style",
            "style_prompt_suffix": ", digital art, concept art, detailed, high resolution",
            "negative_prompt": "traditional media, low quality, blurry",
            "guidance_scale": 8.0,
        },
        "Portrait": {
            "description": "Portrait photography style",
            "style_prompt_suffix": ", portrait, professional photography, detailed face, high quality",
            "negative_prompt": "full body, landscape, low quality, blurry",
            "guidance_scale": 7.5,
        }
    })
    
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
    
    def validate_resolution(self, width: int, height: int, divisor: int = 8) -> Tuple[int, int]:
        """Ensure resolution dimensions are divisible by specified divisor for AI models"""
        def round_to_divisor(value: int, div: int) -> int:
            return ((value + div - 1) // div) * div
        
        return (round_to_divisor(width, divisor), round_to_divisor(height, divisor))
    
    def get_validated_resolution(self, resolution_name: str, model_type: str = None) -> Tuple[int, int]:
        """Get resolution with validation based on model requirements"""
        width, height = self.get_resolution(resolution_name)
        
        # FLUX models require dimensions divisible by 16
        if model_type in ["flux", "flux_lora"]:
            return self.validate_resolution(width, height, divisor=16)
        
        # Default: divisible by 8 for most AI models
        return self.validate_resolution(width, height, divisor=8)
    
    def get_validated_resolution_for_model(self, resolution_name: str, model_name: str) -> Tuple[int, int]:
        """Get resolution validated for specific model"""
        # Determine model type from model name or info
        model_type = None
        
        if model_name in self.SUPPORTED_IMAGE_MODELS:
            model_type = self.SUPPORTED_IMAGE_MODELS[model_name].get('type')
        elif "flux" in model_name.lower():
            model_type = "flux"
        
        return self.get_validated_resolution(resolution_name, model_type)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration for debugging"""
        return {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "server": {
                "host": self.GRADIO_HOST,
                "port": self.GRADIO_PORT,
                "share": self.GRADIO_SHARE,
                "theme": self.GRADIO_THEME
            },
            "models": {
                "default_image": self.DEFAULT_IMAGE_MODEL,
                "default_video": self.DEFAULT_MODEL,
                "default_quality": self.DEFAULT_QUALITY,
                "default_resolution": self.DEFAULT_RESOLUTION
            },
            "directories": {
                "output": self.OUTPUT_DIR,
                "temp": self.TEMP_DIR,
                "cache": self.MODEL_CACHE_DIR
            },
            "performance": {
                "max_concurrent": self.MAX_CONCURRENT_GENERATIONS,
                "cleanup_temp": self.CLEANUP_TEMP_FILES,
                "auto_optimize": self.AUTO_OPTIMIZE_SETTINGS
            },
            "api_tokens": {
                "hf_token_set": bool(os.getenv("HF_TOKEN")),
                "civitai_token_set": bool(os.getenv("CIVITAI_API_TOKEN"))
            }
        }


# Global config instance
config = Config() 