"""
Image processing utilities for Content Creator
"""
import os
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

from ..config import config

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing operations for video generation"""
    
    def __init__(self):
        """Initialize image processor"""
        self.supported_formats = config.SUPPORTED_IMAGE_FORMATS
        self.max_size = config.MAX_IMAGE_SIZE
        logger.debug("Image processor initialized")
    
    def validate_image(self, image_path: Union[str, Path]) -> bool:
        """Validate if an image file is supported"""
        try:
            path = Path(image_path)
            
            if not path.exists():
                logger.warning(f"Image file does not exist: {image_path}")
                return False
            
            if path.suffix.lower() not in self.supported_formats:
                logger.warning(f"Unsupported image format: {path.suffix}")
                return False
            
            with Image.open(path) as img:
                img.verify()
            
            return True
            
        except Exception as e:
            logger.warning(f"Invalid image file {image_path}: {e}")
            return False
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[Image.Image]:
        """Load an image from file"""
        try:
            if not self.validate_image(image_path):
                return None
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img.copy()
                
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def resize_image(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize an image maintaining aspect ratio"""
        try:
            image_copy = image.copy()
            image_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            new_image = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_size[0] - image_copy.width) // 2
            paste_y = (target_size[1] - image_copy.height) // 2
            new_image.paste(image_copy, (paste_x, paste_y))
            
            return new_image
                
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image


# Global image processor instance
image_processor = ImageProcessor() 