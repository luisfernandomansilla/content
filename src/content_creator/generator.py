"""
Main video generator for Content Creator
"""
import os
import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import config
from .models.model_manager import model_manager
from .utils.hardware import hardware_detector
from .utils.image_utils import image_processor
from .utils.video_utils import video_processor

logger = logging.getLogger(__name__)

if not TORCH_AVAILABLE:
    logger.warning("PyTorch not available, video generation will use placeholder mode")


class VideoGenerator:
    """Main video generation class"""
    
    def __init__(self):
        """Initialize video generator"""
        self.model_manager = model_manager
        self.hardware_detector = hardware_detector
        self.image_processor = image_processor
        self.video_processor = video_processor
        
        # Log hardware info
        self.hardware_detector.log_hardware_info()
        
        logger.info("Video generator initialized")
    
    def _get_auth_token(self) -> str:
        """Get authentication token for gated models"""
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                # Log partial token for debugging (first 10 chars + asterisks)
                token_preview = f"{token[:10]}***{token[-4:]}" if len(token) > 14 else "***"
                logger.info(f"✅ HF Token found: {token_preview}")
                logger.info(f"Token length: {len(token)} characters")
                return token
            else:
                logger.warning("❌ No HF authentication token found - some models may not be accessible")
                logger.info("💡 To fix: run 'huggingface-cli login' or set HF_TOKEN environment variable")
                return None
        except ImportError:
            logger.warning("❌ huggingface_hub not available, some models may not load")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting HF token: {e}")
            return None
    
    def generate(
        self,
        prompt: str,
        model_name: str = None,
        style: str = None,
        duration: int = None,
        resolution: str = None,
        fps: int = None,
        output_format: str = None,
        quality: str = None,
        reference_images: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> Optional[str]:
        """Generate a video from the given parameters
        
        Args:
            prompt: Text prompt for video generation
            model_name: Name of the model to use
            style: Video style
            duration: Video duration in seconds
            resolution: Video resolution
            fps: Frames per second
            output_format: Output video format
            quality: Quality preset
            reference_images: List of reference image paths
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to generated video or None if failed
        """
        try:
            # Use default values if not provided
            model_name = model_name or config.DEFAULT_MODEL
            style = style or config.DEFAULT_STYLE
            duration = duration or config.DEFAULT_DURATION
            resolution = resolution or config.DEFAULT_RESOLUTION
            fps = fps or config.DEFAULT_FRAME_RATE
            output_format = output_format or config.DEFAULT_OUTPUT_FORMAT
            quality = quality or config.DEFAULT_QUALITY
            
            if progress_callback:
                progress_callback("Initializing video generation...")
            
            # Validate inputs
            if not self._validate_inputs(prompt, model_name, style, duration, resolution):
                return None
            
            # Get model info and check if available
            model_info = self.model_manager.get_model_info(model_name)
            if not model_info:
                # Try to search for the model on Hugging Face
                if progress_callback:
                    progress_callback(f"Model {model_name} not found, searching on Hugging Face...")
                
                search_results = self.model_manager.search_models(model_name, limit=1)
                if search_results:
                    model_info = search_results[0]
                    logger.info(f"Found model on Hugging Face: {model_name}")
                else:
                    logger.error(f"Model not found anywhere: {model_name}")
                    return None
            
            # Auto-download model if it's from Hugging Face and not available locally
            if model_info.get('source') == 'huggingface':
                if not self.model_manager.is_model_available(model_name):
                    if progress_callback:
                        progress_callback(f"Downloading model {model_name} from Hugging Face...")
                    
                    model_path = self.model_manager.download_model(
                        model_name, 
                        model_info.get('model_id'),
                        progress_callback=progress_callback
                    )
                    
                    if not model_path:
                        logger.error(f"Failed to download model: {model_name}")
                        return None
                    
                    if progress_callback:
                        progress_callback(f"Model {model_name} downloaded successfully!")
                else:
                    if progress_callback:
                        progress_callback(f"Model {model_name} already available locally")
            
            # Process reference images if provided
            processed_images = []
            if reference_images:
                if progress_callback:
                    progress_callback("Processing reference images...")
                
                for img_path in reference_images:
                    img = self.image_processor.load_image(img_path)
                    if img:
                        # Resize to match target resolution (validated to be divisible by 8)
                        target_size = config.get_validated_resolution(resolution)
                        processed_img = self.image_processor.resize_image(img, target_size)
                        processed_images.append(processed_img)
            
            # Generate video
            if progress_callback:
                progress_callback("Generating video...")
            
            video_frames = self._generate_video_frames(
                prompt=prompt,
                model_name=model_name,
                model_info=model_info,
                style=style,
                duration=duration,
                resolution=resolution,
                quality=quality,
                reference_images=processed_images,
                progress_callback=progress_callback
            )
            
            if not video_frames:
                logger.error("Failed to generate video frames")
                return None
            
            # Create output path
            timestamp = int(time.time())
            output_filename = f"video_{timestamp}.{output_format.lower()}"
            output_path = Path(config.OUTPUT_DIR) / output_filename
            
            if progress_callback:
                progress_callback("Saving video...")
            
            # Save video
            success = self.video_processor.create_video_from_frames(
                frames=video_frames,
                output_path=str(output_path),
                fps=fps,
                format=output_format
            )
            
            if success:
                if progress_callback:
                    progress_callback(f"Video generated successfully: {output_path}")
                
                logger.info(f"Video generated: {output_path}")
                return str(output_path)
            else:
                logger.error("Failed to save video")
                return None
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            if progress_callback:
                progress_callback(f"Error: {e}")
            return None
    
    def _validate_inputs(
        self, 
        prompt: str, 
        model_name: str, 
        style: str, 
        duration: int, 
        resolution: str
    ) -> bool:
        """Validate input parameters"""
        if not prompt or not prompt.strip():
            logger.error("Prompt cannot be empty")
            return False
        
        if duration < config.MIN_DURATION or duration > config.MAX_DURATION:
            logger.error(f"Duration must be between {config.MIN_DURATION} and {config.MAX_DURATION} seconds")
            return False
        
        if style not in config.VIDEO_STYLES:
            logger.error(f"Invalid style: {style}")
            return False
        
        if resolution not in config.RESOLUTIONS:
            logger.error(f"Invalid resolution: {resolution}")
            return False
        
        return True
    
    def _generate_video_frames(
        self,
        prompt: str,
        model_name: str,
        model_info: Dict[str, Any],
        style: str,
        duration: int,
        resolution: str,
        quality: str,
        reference_images: List[Image.Image],
        progress_callback: Optional[callable] = None
    ) -> Optional[List[np.ndarray]]:
        """Generate video frames using the specified model"""
        try:
            # Check if torch is available
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available, using placeholder video")
                if progress_callback:
                    progress_callback("PyTorch not available, generating placeholder...")
                return self._generate_placeholder_frames(prompt, model_name, style, duration, resolution, progress_callback)
            
            # Get hardware optimizations
            optimizations = self.hardware_detector.optimize_for_generation(
                model_name, resolution, duration
            )
            
            # Get style info
            style_info = config.get_style_info(style)
            
            # Get quality settings
            quality_settings = config.get_quality_settings(quality)
            
            # Get resolution (validated to be divisible by 8)
            width, height = config.get_validated_resolution(resolution)
            
            # Calculate number of frames
            fps = config.DEFAULT_FRAME_RATE
            num_frames = duration * fps
            
            # Enhanced prompt with style
            enhanced_prompt = prompt + style_info.get('style_prompt_suffix', '')
            
            if progress_callback:
                progress_callback(f"Loading video model {model_name}...")
            
            # Load the appropriate pipeline for video generation
            pipeline = self._load_video_pipeline(model_name, model_info, progress_callback)
            
            if not pipeline:
                logger.error(f"Failed to load video pipeline for {model_name}")
                logger.warning("Falling back to placeholder video generation")
                return self._generate_placeholder_frames(prompt, model_name, style, duration, resolution, progress_callback)
            
            if progress_callback:
                progress_callback(f"Generating video at {width}x{height} for {duration}s...")
            
            # Generate video based on model type
            frames = self._generate_real_video_frames(
                pipeline=pipeline,
                prompt=enhanced_prompt,
                model_name=model_name,
                model_info=model_info,
                width=width,
                height=height,
                num_frames=num_frames,
                reference_images=reference_images,
                progress_callback=progress_callback
            )
            
            if frames:
                logger.info(f"Generated {len(frames)} frames with model {model_name}")
                return frames
            else:
                logger.warning("Real video generation failed, falling back to placeholder")
                return self._generate_placeholder_frames(prompt, model_name, style, duration, resolution, progress_callback)
            
        except Exception as e:
            logger.error(f"Error generating video frames: {e}")
            logger.warning("Falling back to placeholder video generation")
            return self._generate_placeholder_frames(prompt, model_name, style, duration, resolution, progress_callback)
    
    def _load_video_pipeline(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ):
        """Load the appropriate video generation pipeline"""
        try:
            import torch
            from diffusers import (
                StableVideoDiffusionPipeline,
                AnimateDiffPipeline,
                DiffusionPipeline,
                TextToVideoSDPipeline,
                VideoToVideoSDPipeline
            )
            
            model_id = model_info.get('model_id', model_name)
            model_type = model_info.get('type', 'video')
            
            # Determine device
            device = self.hardware_detector.get_optimal_device()
            
            # Set torch dtype based on hardware
            if self.hardware_detector.hardware_type == "apple_silicon":
                torch_dtype = torch.float16
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            if progress_callback:
                progress_callback(f"Loading {model_type} pipeline...")
            
            # Get authentication token for gated models
            token = self._get_auth_token()
            
            # Load pipeline based on model type
            if "animatediff" in model_name.lower() or model_type == "animatediff":
                # AnimateDiff models
                try:
                    pipeline = AnimateDiffPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None,
                        token=token
                    )
                except Exception as e:
                    logger.warning(f"Failed to load AnimateDiff with authentication: {e}")
                    # Try without authentication
                    pipeline = AnimateDiffPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None
                    )
                
            elif "stable-video-diffusion" in model_name.lower() or model_type == "svd":
                # Stable Video Diffusion
                try:
                    pipeline = StableVideoDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None,
                        token=token
                    )
                except Exception as e:
                    logger.warning(f"Failed to load SVD with authentication: {e}")
                    pipeline = StableVideoDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None
                    )
                
            elif "text2video" in model_name.lower() or model_type == "text2video":
                # Text2Video models
                try:
                    pipeline = TextToVideoSDPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None,
                        token=token
                    )
                except Exception as e:
                    logger.warning(f"Failed to load Text2Video with authentication: {e}")
                    pipeline = TextToVideoSDPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None
                    )
                
            else:
                # Try generic video pipeline
                try:
                    try:
                        pipeline = DiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            use_safetensors=True,
                            variant="fp16" if torch_dtype == torch.float16 else None,
                            token=token
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load DiffusionPipeline with authentication: {e}")
                        pipeline = DiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            use_safetensors=True,
                            variant="fp16" if torch_dtype == torch.float16 else None
                        )
                except:
                    # Fallback to text2video
                    try:
                        pipeline = TextToVideoSDPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            use_safetensors=True,
                            variant="fp16" if torch_dtype == torch.float16 else None,
                            token=token
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load fallback Text2Video with authentication: {e}")
                        pipeline = TextToVideoSDPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            use_safetensors=True,
                            variant="fp16" if torch_dtype == torch.float16 else None
                        )
            
            # Move to device
            pipeline = pipeline.to(device)
            
            # Enable optimizations
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing()
            
            if hasattr(pipeline, 'enable_memory_efficient_attention') and torch.cuda.is_available():
                try:
                    pipeline.enable_memory_efficient_attention()
                except:
                    pass
            
            # Enable xformers if available
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except:
                    pass
            
            if progress_callback:
                progress_callback(f"Video pipeline loaded successfully on {device}")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading video pipeline for {model_name}: {e}")
            if progress_callback:
                progress_callback(f"Error loading video model: {e}")
            return None
    
    def _generate_real_video_frames(
        self,
        pipeline,
        prompt: str,
        model_name: str,
        model_info: Dict[str, Any],
        width: int,
        height: int,
        num_frames: int,
        reference_images: List[Image.Image],
        progress_callback: Optional[callable] = None
    ) -> Optional[List[np.ndarray]]:
        """Generate real video frames using the loaded pipeline"""
        try:
            import torch
            
            if progress_callback:
                progress_callback(f"Generating {num_frames} frames...")
            
            # Set up generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "guidance_scale": 7.5,
                "num_inference_steps": 20,  # Lower for faster generation
            }
            
            # Handle different pipeline types
            model_type = model_info.get('type', 'video')
            
            if model_type == "svd" and reference_images:
                # Stable Video Diffusion requires an image input
                generation_kwargs["image"] = reference_images[0]
                generation_kwargs.pop("prompt", None)  # SVD doesn't use text prompts
                
            elif "animatediff" in model_name.lower():
                # AnimateDiff specific parameters
                generation_kwargs["num_inference_steps"] = 25
                
            # Generate the video
            with torch.inference_mode():
                result = pipeline(**generation_kwargs)
            
            # Extract frames from result
            if hasattr(result, 'frames') and len(result.frames) > 0:
                # Convert PIL Images to numpy arrays
                frames = []
                for frame in result.frames[0]:  # First sequence
                    frame_array = np.array(frame)
                    frames.append(frame_array)
                
                if progress_callback:
                    progress_callback(f"Generated {len(frames)} frames successfully!")
                
                return frames
                
            elif hasattr(result, 'images'):
                # Some models return images instead of frames
                frames = []
                for img in result.images:
                    frame_array = np.array(img)
                    frames.append(frame_array)
                
                if progress_callback:
                    progress_callback(f"Generated {len(frames)} frames successfully!")
                
                return frames
            else:
                logger.error("Unexpected result format from video pipeline")
                return None
                
        except Exception as e:
            logger.error(f"Error in real video generation: {e}")
            if progress_callback:
                progress_callback(f"Video generation error: {e}")
            return None
    
    def _generate_placeholder_frames(
        self,
        prompt: str,
        model_name: str,
        style: str,
        duration: int,
        resolution: str,
        progress_callback: Optional[callable] = None
    ) -> List[np.ndarray]:
        """Generate placeholder frames for demonstration"""
        try:
            # Get resolution (validated to be divisible by 8)
            width, height = config.get_validated_resolution(resolution)
            
            # Calculate number of frames
            fps = config.DEFAULT_FRAME_RATE
            num_frames = duration * fps
            
            if progress_callback:
                progress_callback(f"Generating {num_frames} placeholder frames...")
            
            frames = []
            for i in range(num_frames):
                frame = self._create_placeholder_frame(
                    width, height, i, num_frames, prompt
                )
                frames.append(frame)
                
                if progress_callback and i % 5 == 0:
                    progress = (i + 1) / num_frames * 100
                    progress_callback(f"Generated frame {i+1}/{num_frames} ({progress:.1f}%)")
            
            logger.info(f"Generated {len(frames)} placeholder frames")
            return frames
            
        except Exception as e:
            logger.error(f"Error generating placeholder frames: {e}")
            return []
    
    def _create_placeholder_frame(
        self, 
        width: int, 
        height: int, 
        frame_index: int, 
        total_frames: int,
        prompt: str
    ) -> np.ndarray:
        """Create a placeholder frame for demonstration
        
        In a real implementation, this would be replaced with actual model inference
        """
        # Create a gradient that changes over time
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a time-based color gradient
        t = frame_index / total_frames
        
        # RGB values that change over time
        r = int(128 + 127 * np.sin(t * 2 * np.pi))
        g = int(128 + 127 * np.sin(t * 2 * np.pi + 2))
        b = int(128 + 127 * np.sin(t * 2 * np.pi + 4))
        
        # Create gradient
        for y in range(height):
            for x in range(width):
                # Gradient based on position and time
                gradient_r = int(r * (x / width))
                gradient_g = int(g * (y / height))
                gradient_b = int(b * ((x + y) / (width + height)))
                
                frame[y, x] = [gradient_r, gradient_g, gradient_b]
        
        # Add some moving elements
        center_x = int(width * (0.5 + 0.3 * np.sin(t * 4 * np.pi)))
        center_y = int(height * (0.5 + 0.3 * np.cos(t * 4 * np.pi)))
        
        # Draw a circle that moves
        cv2_available = True
        try:
            import cv2
            cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), -1)
        except ImportError:
            cv2_available = False
        
        return frame
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models"""
        return self.model_manager.get_available_models()
    
    def search_models(
        self, 
        query: str, 
        platforms: List[str] = None, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for models across different platforms"""
        return self.model_manager.search_models(
            query=query, 
            platforms=platforms, 
            limit=limit
        )
    
    def get_recommended_models(self) -> List[str]:
        """Get recommended models for current hardware"""
        return self.model_manager.get_recommended_models()
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information"""
        return self.hardware_detector.get_device_info()


# Global generator instance
generator = VideoGenerator() 