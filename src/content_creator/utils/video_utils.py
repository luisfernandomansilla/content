"""
Video processing utilities for Content Creator
"""
import os
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import cv2
import imageio
from PIL import Image

from ..config import config

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self):
        """Initialize video processor"""
        self.output_formats = config.OUTPUT_FORMATS
        logger.debug("Video processor initialized")
    
    def create_video_from_frames(
        self, 
        frames: List[np.ndarray], 
        output_path: str,
        fps: int = 24,
        format: str = "MP4"
    ) -> bool:
        """Create video from list of frames
        
        Args:
            frames: List of frame arrays
            output_path: Output video path
            fps: Frames per second
            format: Output format
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not frames:
                logger.error("No frames provided for video creation")
                return False
            
            # Get format info
            format_info = self.output_formats.get(format, self.output_formats["MP4"])
            
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format == "GIF":
                # Create GIF
                pil_frames = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        pil_frame = Image.fromarray(frame.astype(np.uint8))
                        pil_frames.append(pil_frame)
                
                pil_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=int(1000 / fps),
                    loop=0
                )
                
            else:
                # Create video using imageio
                with imageio.get_writer(
                    output_path, 
                    fps=fps, 
                    codec=format_info.get("codec", "libx264")
                ) as writer:
                    for frame in frames:
                        if isinstance(frame, np.ndarray):
                            writer.append_data(frame.astype(np.uint8))
            
            logger.info(f"Successfully created video: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            return False
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video information"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            }
            
            cap.release()
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}


# Global video processor instance
video_processor = VideoProcessor() 