"""
Content Creator - AI-powered video and image generation tool
"""

__version__ = "1.0.0"
__author__ = "Content Creator Team"
__description__ = "AI-powered video and image generation using vLLM and Gradio"

from .generator import VideoGenerator, generator
from .image_generator import ImageGenerator, image_generator
from .config import Config
from .utils.hardware import HardwareDetector

__all__ = ["VideoGenerator", "generator", "ImageGenerator", "image_generator", "Config", "HardwareDetector"] 