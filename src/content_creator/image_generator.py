"""
Image generator for Content Creator
"""
import os
import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import hashlib
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, image generation will use placeholder mode")

from .config import config
from .models.model_manager import model_manager
from .utils.hardware import hardware_detector
from .utils.image_utils import image_processor

# Core ML imports
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline,
    FluxPipeline,
    AutoPipelineForText2Image,
    DiffusionPipeline
)

# Additional imports for long prompt handling
try:
    from compel import Compel, ReturnedEmbeddingsType
    COMPEL_AVAILABLE = True
except ImportError:
    COMPEL_AVAILABLE = False
    logging.warning("Compel not available - long prompt support limited")

logger = logging.getLogger(__name__)


def chunk_prompt(prompt: str, max_length: int = 75) -> List[str]:
    """Split a long prompt into chunks that fit within token limits"""
    # Simple word-based chunking
    words = prompt.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        # Rough token estimate: ~1.3 tokens per word for English
        word_tokens = len(word.split()) + 1
        
        if current_length + word_tokens > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_tokens
        else:
            current_chunk.append(word)
            current_length += word_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def prioritize_prompt_parts(prompt: str) -> str:
    """Prioritize important parts of the prompt for token-limited models"""
    # Extract key elements and prioritize them
    important_keywords = [
        'masterpiece', 'best quality', 'high quality', 'detailed', 'high resolution',
        'cinematic', '4k', '8k', 'ultra detailed', 'photorealistic', 'realistic'
    ]
    
    # Split prompt into parts
    parts = [part.strip() for part in prompt.split(',')]
    
    # Separate quality terms from descriptive terms
    quality_parts = []
    descriptive_parts = []
    character_parts = []
    
    for part in parts:
        part_lower = part.lower()
        if any(keyword in part_lower for keyword in important_keywords):
            quality_parts.append(part)
        elif any(char_term in part_lower for char_term in ['girl', 'boy', 'woman', 'man', 'person', 'character']):
            character_parts.append(part)
        else:
            descriptive_parts.append(part)
    
    # Rebuild prompt with priority order: character -> key descriptive -> quality
    prioritized_parts = character_parts[:3] + descriptive_parts[:8] + quality_parts[:3]
    
    return ", ".join(prioritized_parts)


def enhance_prompt_for_model(prompt: str, model_type: str) -> str:
    """Enhance prompt based on model capabilities"""
    if model_type == "flux" or model_type == "flux_lora":
        # FLUX has T5 text encoder - can handle longer prompts
        return prompt
    elif model_type in ["stable_diffusion", "diffusion", "lora", "checkpoint"]:
        # Use prioritized version for CLIP-based models
        word_count = len(prompt.split())
        estimated_tokens = int(word_count * 1.3)
        
        if estimated_tokens > 75:  # Close to CLIP limit
            logger.warning(f"🔤 Long prompt detected ({word_count} words, ~{estimated_tokens} tokens). Prioritizing key elements...")
            return prioritize_prompt_parts(prompt)
        return prompt
    else:
        # For unknown model types, assume CLIP limitation
        word_count = len(prompt.split())
        estimated_tokens = int(word_count * 1.3)
        
        if estimated_tokens > 75:
            logger.warning(f"🔤 Long prompt detected for unknown model type '{model_type}'. Prioritizing key elements...")
            return prioritize_prompt_parts(prompt)
        return prompt


class ImageGenerator:
    """Main image generation class"""
    
    def __init__(self):
        """Initialize image generator"""
        self.model_manager = model_manager
        self.hardware_detector = hardware_detector
        self.image_processor = image_processor
        
        # Pipeline cache to avoid reloading models
        self._pipeline_cache = {}
        self._current_model = None
        self._current_pipeline = None
        
        logger.info("Image generator initialized with pipeline caching")
    
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
        resolution: str = None,
        output_format: str = None,
        quality: str = None,
        reference_images: Optional[List[str]] = None,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 28,
        seed: Optional[int] = None,
        progress_callback: Optional[callable] = None,
        preferred_base_model: str = None
    ) -> Optional[str]:
        """Generate an image from the given parameters
        
        Args:
            prompt: Text prompt for image generation
            model_name: Name of the model to use
            style: Image style
            resolution: Image resolution
            output_format: Output image format
            quality: Quality preset
            reference_images: List of reference image paths
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            progress_callback: Optional callback for progress updates
            preferred_base_model: Preferred base model for LoRA
            
        Returns:
            Path to generated image or None if failed
        """
        try:
            # Use default values if not provided
            model_name = model_name or config.DEFAULT_IMAGE_MODEL
            style = style or config.DEFAULT_STYLE
            resolution = resolution or config.DEFAULT_RESOLUTION
            output_format = output_format or "PNG"
            quality = quality or config.DEFAULT_QUALITY
            
            if progress_callback:
                progress_callback("Initializing image generation...")
            
            # Validate inputs
            if not self._validate_inputs(prompt, model_name, style, resolution):
                return None
            
            # Get model info and check if available
            model_info = self._get_image_model_info(model_name)
            if not model_info:
                # Try to search for the model on Hugging Face
                if progress_callback:
                    progress_callback(f"Model {model_name} not found, searching on Hugging Face...")
                
                search_results = self._search_image_models(model_name, limit=1)
                if search_results:
                    model_info = search_results[0]
                    logger.info(f"Found image model on Hugging Face: {model_name}")
                else:
                    logger.error(f"Image model not found anywhere: {model_name}")
                    return None
            
            # Auto-download model if needed
            if not self._is_image_model_available(model_name):
                if progress_callback:
                    progress_callback(f"Downloading image model {model_name} from Hugging Face...")
                
                download_success = self._download_image_model(
                    model_name, 
                    model_info,
                    progress_callback=progress_callback
                )
                
                if not download_success:
                    logger.error(f"Failed to download image model: {model_name}")
                    return None
                
                if progress_callback:
                    progress_callback(f"Image model {model_name} downloaded successfully!")
            else:
                if progress_callback:
                    progress_callback(f"Image model {model_name} already available locally")
            
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
            
            # Generate image
            if progress_callback:
                progress_callback("Generating image...")
            
            generated_image = self._generate_image(
                prompt=prompt,
                model_name=model_name,
                model_info=model_info,
                style=style,
                resolution=resolution,
                quality=quality,
                reference_images=processed_images,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                progress_callback=progress_callback,
                preferred_base_model=preferred_base_model
            )
            
            if not generated_image:
                logger.error("Failed to generate image")
                return None
            
            # Create output path
            timestamp = int(time.time())
            output_filename = f"image_{timestamp}.{output_format.lower()}"
            output_path = Path(config.OUTPUT_DIR) / output_filename
            
            if progress_callback:
                progress_callback("Saving image...")
            
            # Save image
            try:
                os.makedirs(output_path.parent, exist_ok=True)
                generated_image.save(output_path, format=output_format)
                
                if progress_callback:
                    progress_callback(f"Image generated successfully: {output_path}")
                
                logger.info(f"Image generated: {output_path}")
                # Return absolute path to ensure Gradio can access the image
                return str(output_path.absolute())
                
            except Exception as e:
                logger.error(f"Failed to save image: {e}")
                return None
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            if progress_callback:
                progress_callback(f"Error: {e}")
            return None
    
    def _validate_inputs(
        self, 
        prompt: str, 
        model_name: str, 
        style: str, 
        resolution: str
    ) -> bool:
        """Validate input parameters"""
        if not prompt or not prompt.strip():
            logger.error("Prompt cannot be empty")
            return False
        
        if style not in config.VIDEO_STYLES:
            logger.error(f"Invalid style: {style}")
            return False
        
        if resolution not in config.RESOLUTIONS:
            logger.error(f"Invalid resolution: {resolution}")
            return False
        
        return True
    
    def _get_image_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get image model information"""
        # First check supported image models
        if model_name in config.SUPPORTED_IMAGE_MODELS:
            return {
                'name': model_name,
                'source': 'default',
                **config.SUPPORTED_IMAGE_MODELS[model_name]
            }
        
        # Then check downloaded models from model manager
        available_models = self.model_manager.get_available_models(include_online=False)
        
        # Direct name match
        if model_name in available_models:
            return available_models[model_name]
        
        # Try name normalization for LoRA models - handle common variations
        normalized_variations = self._get_model_name_variations(model_name)
        
        for variation in normalized_variations:
            if variation in available_models:
                logger.info(f"✅ Found model using name variation: '{variation}' for requested '{model_name}'")
                return available_models[variation]
        
        # Search by partial name match for LoRA models
        for available_name, model_info in available_models.items():
            # Check if it's likely the same model (case-insensitive partial match)
            if self._is_likely_same_model(model_name, available_name):
                logger.info(f"✅ Found model by partial match: '{available_name}' for requested '{model_name}'")
                return model_info
        
        return None
    
    def _get_model_name_variations(self, model_name: str) -> List[str]:
        """Generate common name variations for model lookup"""
        variations = [model_name]
        
        # Handle common LoRA naming patterns
        base_name = model_name
        
        # Remove common suffixes
        if base_name.endswith(" (LoRA)"):
            base_name = base_name[:-7]
            variations.append(base_name)
        
        if base_name.endswith("(LoRA)"):
            base_name = base_name[:-6]
            variations.append(base_name)
        
        # Handle underscore vs space vs dash variations
        underscore_version = base_name.replace(" ", "_").replace("-", "_")
        space_version = base_name.replace("_", " ").replace("-", " ")
        dash_version = base_name.replace("_", "-").replace(" ", "-")
        
        variations.extend([underscore_version, space_version, dash_version])
        
        # Handle common patterns in Civitai model names
        if "_-_" in base_name:
            variations.append(base_name.replace("_-_", " - "))
        if " - " in base_name:
            variations.append(base_name.replace(" - ", "_-_"))
        
        # Remove duplicates while preserving order
        unique_variations = []
        for var in variations:
            if var not in unique_variations:
                unique_variations.append(var)
        
        return unique_variations
    
    def _is_likely_same_model(self, requested_name: str, available_name: str) -> bool:
        """Check if two model names likely refer to the same model"""
        # Normalize both names for comparison
        def normalize_for_comparison(name: str) -> str:
            return name.lower().replace("_", "").replace("-", "").replace(" ", "").replace("(lora)", "")
        
        normalized_requested = normalize_for_comparison(requested_name)
        normalized_available = normalize_for_comparison(available_name)
        
        # Check if one contains the other (for partial matches)
        if len(normalized_requested) > 5 and len(normalized_available) > 5:
            return (normalized_requested in normalized_available or 
                    normalized_available in normalized_requested)
        
        return False
    
    def _search_image_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for image models (placeholder - could extend to search HF)"""
        # Simple search in our supported models
        results = []
        query_lower = query.lower()
        
        for name, info in config.SUPPORTED_IMAGE_MODELS.items():
            if (query_lower in name.lower() or 
                query_lower in info.get('description', '').lower()):
                results.append({
                    'name': name,
                    'source': 'default',
                    **info
                })
        
        return results[:limit]
    
    def _is_image_model_available(self, model_name: str) -> bool:
        """Check if image model is available locally"""
        try:
            # First check if we can get model info (this includes downloaded models)
            model_info = self._get_image_model_info(model_name)
            if not model_info:
                return False
            
            # If it's a local file (like LoRA), check if the file exists
            if model_info.get('source') == 'local_file':
                model_path = model_info.get('path')
                if model_path:
                    from pathlib import Path
                    if Path(model_path).exists():
                        logger.info(f"✅ Local model file found: {model_path}")
                        return True
                    else:
                        logger.warning(f"❌ Local model file not found: {model_path}")
                        return False
            
            # If it's a built-in supported model, check HuggingFace cache
            if model_name in config.SUPPORTED_IMAGE_MODELS:
                model_id = model_info.get('model_id', model_name)
                
                # Try to check if model is cached locally
                try:
                    from huggingface_hub import cached_assets_path
                    from pathlib import Path
                    
                    # Check if model files exist in cache
                    cache_path = cached_assets_path(library_name="diffusers", namespace="models--" + model_id.replace('/', '--'))
                    
                    if cache_path and Path(cache_path).exists():
                        # Check for some essential files
                        essential_files = ["model_index.json", "config.json"]
                        for file in essential_files:
                            if any(Path(cache_path).rglob(file)):
                                logger.info(f"✅ HuggingFace cached model found: {model_id}")
                                return True
                    
                    logger.info(f"❌ HuggingFace model not cached: {model_id}")
                    return False
                    
                except ImportError:
                    # If huggingface_hub is not available, assume not available
                    logger.warning("❌ huggingface_hub not available, assuming model not cached")
                    return False
            
            # If it's a downloaded HuggingFace model, check if it's in our downloaded models
            if model_info.get('source') in ['huggingface_downloaded', 'downloaded']:
                # Check if model directory exists
                model_path = model_info.get('path')
                if model_path:
                    from pathlib import Path
                    if Path(model_path).exists():
                        logger.info(f"✅ Downloaded HuggingFace model found: {model_path}")
                        return True
                    else:
                        logger.warning(f"❌ Downloaded HuggingFace model path not found: {model_path}")
                        return False
            
            # Default: not available
            logger.warning(f"❌ Model not available locally: {model_name}")
            return False
                
        except Exception as e:
            logger.debug(f"Error checking model availability for {model_name}: {e}")
            return False
    
    def _download_image_model(
        self, 
        model_name: str, 
        model_info: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> bool:
        """Download image model if needed"""
        try:
            # Use the existing model manager for downloads
            model_id = model_info.get('model_id', model_name)
            
            result = self.model_manager.download_model(
                model_name=model_name,
                model_id=model_id,
                progress_callback=progress_callback
            )
            
            return result is not None
            
        except Exception as e:
            logger.error(f"Error downloading image model {model_name}: {e}")
            return False
    
    def _generate_image(
        self,
        prompt: str,
        model_name: str,
        model_info: Dict[str, Any],
        style: str,
        resolution: str,
        quality: str,
        reference_images: List[Image.Image],
        negative_prompt: str,
        guidance_scale: float,
        num_inference_steps: int,
        seed: Optional[int],
        progress_callback: Optional[callable] = None,
        preferred_base_model: str = None
    ) -> Optional[Image.Image]:
        """Generate an image using the specified model and parameters"""
        try:
            if progress_callback:
                progress_callback(f"Preparing generation with {model_name}...")
            
            # Get style information
            style_info = config.IMAGE_STYLES.get(style, config.IMAGE_STYLES.get("None", {}))
            
            # Get quality settings
            quality_settings = config.get_quality_settings(quality)
            if quality_settings:
                # Override inference steps and guidance scale from quality settings if they're defaults
                if num_inference_steps == 28:  # Default value
                    num_inference_steps = quality_settings.get('inference_steps', num_inference_steps)
                if guidance_scale == 7.5:  # Default value
                    guidance_scale = quality_settings.get('guidance_scale', guidance_scale)
            
            # Get resolution (validated for the specific model)
            width, height = config.get_validated_resolution_for_model(resolution, model_name)
            
            # Enhanced prompt with style
            enhanced_prompt = prompt + style_info.get('style_prompt_suffix', '')
            
            # Optimize prompt for the specific model type
            model_type = model_info.get('type', 'unknown')
            original_prompt = enhanced_prompt
            enhanced_prompt = enhance_prompt_for_model(enhanced_prompt, model_type)
            
            # Log prompt optimization if it occurred
            if enhanced_prompt != original_prompt:
                logger.info(f"🔤 Prompt optimized for {model_type}:")
                logger.info(f"   Original length: {len(original_prompt.split())} words")
                logger.info(f"   Optimized length: {len(enhanced_prompt.split())} words")
                if progress_callback:
                    progress_callback(f"Optimizing long prompt for {model_type} model...")
            
            if progress_callback:
                progress_callback(f"Loading model {model_name}...")
            
            # Load the appropriate pipeline based on model type (with caching)
            pipeline = self._get_or_load_pipeline(model_name, model_info, progress_callback, preferred_base_model)
            
            if not pipeline:
                logger.error(f"Failed to load pipeline for {model_name}")
                return None
            
            if progress_callback:
                progress_callback(f"Generating image at {width}x{height}...")
            
            # Set up generation parameters with advanced prompt handling
            generation_kwargs = {
                "prompt": enhanced_prompt,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }
            
            # Advanced prompt processing for CLIP-limited models
            if model_type in ["stable_diffusion", "diffusion"] and COMPEL_AVAILABLE:
                try:
                    # Use Compel for better prompt handling on CLIP-based models
                    if hasattr(pipeline, 'text_encoder'):
                        compel = Compel(
                            tokenizer=pipeline.tokenizer,
                            text_encoder=pipeline.text_encoder,
                            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                            requires_pooled=False
                        )
                        
                        # Generate embeddings for the prompt
                        prompt_embeds = compel(enhanced_prompt)
                        generation_kwargs["prompt_embeds"] = prompt_embeds
                        generation_kwargs.pop("prompt", None)  # Remove text prompt since we're using embeddings
                        
                        logger.info("✅ Using Compel for advanced prompt processing")
                        
                        # Handle negative prompt with Compel if available
                        if negative_prompt:
                            negative_embeds = compel(negative_prompt)
                            generation_kwargs["negative_prompt_embeds"] = negative_embeds
                        else:
                            # Use empty embedding for negative prompt
                            negative_embeds = compel("")
                            generation_kwargs["negative_prompt_embeds"] = negative_embeds
                except Exception as e:
                    logger.warning(f"⚠️ Compel processing failed, using standard prompt: {e}")
                    # Fallback to standard prompt processing
                    if negative_prompt:
                        generation_kwargs["negative_prompt"] = negative_prompt
            else:
                # Standard prompt processing for FLUX and other models
                if hasattr(pipeline, 'text_encoder') and negative_prompt:
                    generation_kwargs["negative_prompt"] = negative_prompt
            
            # Set seed for reproducibility
            if seed is not None:
                generator = torch.Generator(device=pipeline.device).manual_seed(seed)
                generation_kwargs["generator"] = generator
            
            # Generate the image
            with torch.inference_mode():
                result = pipeline(**generation_kwargs)
                
            # Extract the image from the result
            if hasattr(result, 'images') and len(result.images) > 0:
                generated_image = result.images[0]
            else:
                generated_image = result
            
            if progress_callback:
                progress_callback("Image generation completed!")
            
            logger.info(f"Generated image with model {model_name}")
            return generated_image
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            if progress_callback:
                progress_callback(f"Error: {e}")
            
            # Fallback to placeholder if real generation fails
            logger.warning("Falling back to placeholder image generation")
            return self._create_placeholder_image(
                width, height, enhanced_prompt, model_name
            )
    
    def _get_or_load_pipeline(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        progress_callback: Optional[callable] = None,
        preferred_base_model: str = None
    ):
        """Get cached pipeline or load new one"""
        # Special handling for LoRA models
        if model_info.get('type') == 'lora':
            return self._handle_local_lora_model(model_name, model_info, progress_callback, preferred_base_model)
        
        # Check if we have this model cached
        cache_key = f"{model_name}_{model_info.get('model_id', model_name)}"
        
        if cache_key in self._pipeline_cache:
            logger.info(f"📋 Using cached pipeline for {model_name}")
            if progress_callback:
                progress_callback(f"Using cached model {model_name}...")
                
            pipeline = self._pipeline_cache[cache_key]
            self._current_model = model_name
            self._current_pipeline = pipeline
            return pipeline
        
        # Load new pipeline
        logger.info(f"🔄 Loading new pipeline for {model_name}")
        pipeline = self._load_pipeline(model_name, model_info, progress_callback)
        
        if pipeline:
            # Cache the pipeline
            self._pipeline_cache[cache_key] = pipeline
            self._current_model = model_name
            self._current_pipeline = pipeline
            logger.info(f"✅ Pipeline cached for {model_name}")
            
        return pipeline
    
    def _handle_local_lora_model(self, model_name: str, model_info: Dict[str, Any], progress_callback: Optional[callable] = None, preferred_base_model: str = None):
        """Handle local LoRA model loading with user-selected base model"""
        try:
            if progress_callback:
                progress_callback(f"Loading LoRA model: {model_name}...")
            
            # Get compatible base models
            from .models.model_manager import model_manager
            base_models = model_manager.get_base_models_for_lora(model_name)
            
            if not base_models:
                logger.error(f"No compatible base models found for LoRA: {model_name}")
                if progress_callback:
                    progress_callback(f"Error: No compatible base models for LoRA {model_name}")
                return None
            
            # Use the user-selected base model if it's compatible, otherwise use the first compatible one
            base_model_name = None
            if preferred_base_model and preferred_base_model in base_models:
                base_model_name = preferred_base_model
                logger.info(f"✅ Using user-selected base model: {preferred_base_model} for LoRA: {model_name}")
            else:
                # Fallback to first compatible base model
                base_model_name = base_models[0]
                logger.warning(f"⚠️ User-selected base model '{preferred_base_model}' not compatible with LoRA {model_name}")
                logger.info(f"📋 Using first compatible base model: {base_model_name}")
            
            if progress_callback:
                progress_callback(f"Using base model: {base_model_name} with LoRA: {model_name}")
            
            logger.info(f"🎭 Loading LoRA {model_name} with base model {base_model_name}")
            
            # Get base model info
            all_models = model_manager.get_available_models()
            base_model_info = all_models.get(base_model_name)
            
            if not base_model_info:
                logger.error(f"Base model info not found: {base_model_name}")
                return None
            
            # Create a unique cache key for this LoRA + base model combination
            cache_key = f"lora_{model_name}_{base_model_name}"
            
            # Check if this combination is already cached
            if cache_key in self._pipeline_cache:
                logger.info(f"📋 Using cached LoRA+base pipeline: {model_name} + {base_model_name}")
                if progress_callback:
                    progress_callback(f"Using cached LoRA combination...")
                    
                pipeline = self._pipeline_cache[cache_key]
                self._current_model = f"{model_name} (LoRA on {base_model_name})"
                self._current_pipeline = pipeline
                return pipeline
            
            # Load base pipeline
            base_pipeline = self._load_pipeline(base_model_name, base_model_info, progress_callback)
            
            if not base_pipeline:
                logger.error(f"Failed to load base pipeline: {base_model_name}")
                return None
            
            # Apply LoRA to the pipeline
            lora_path = model_info.get('path')
            if lora_path and Path(lora_path).exists():
                try:
                    if progress_callback:
                        progress_callback(f"Applying LoRA: {model_name}...")
                    
                    logger.info(f"🔗 Applying LoRA from: {lora_path}")
                    
                    # Load LoRA weights
                    if hasattr(base_pipeline, 'load_lora_weights'):
                        base_pipeline.load_lora_weights(lora_path, adapter_name="lora_adapter")
                        logger.info(f"✅ LoRA applied successfully: {model_name}")
                        
                        if progress_callback:
                            progress_callback(f"LoRA {model_name} applied to {base_model_name}")
                        
                        # Cache the combined pipeline
                        self._pipeline_cache[cache_key] = base_pipeline
                        self._current_model = f"{model_name} (LoRA on {base_model_name})"
                        self._current_pipeline = base_pipeline
                        
                        return base_pipeline
                    else:
                        logger.warning(f"Base model {base_model_name} doesn't support LoRA loading")
                        logger.info(f"Using base model {base_model_name} without LoRA")
                        return base_pipeline
                        
                except Exception as e:
                    logger.warning(f"Failed to apply LoRA {model_name}: {e}")
                    logger.info(f"Using base model {base_model_name} without LoRA")
                    if progress_callback:
                        progress_callback(f"LoRA failed, using base model {base_model_name}")
                    return base_pipeline
            else:
                logger.error(f"LoRA file not found: {lora_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error handling LoRA model {model_name}: {e}")
            return None
    
    def _load_pipeline(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ):
        """Load the appropriate pipeline for the model"""
        try:
            import torch
            from diffusers import (
                AutoPipelineForText2Image,
                StableDiffusionPipeline,
                StableDiffusionXLPipeline,
                FluxPipeline,
                DiffusionPipeline
            )
            
            model_id = model_info.get('model_id', model_name)
            model_type = model_info.get('type', 'diffusion')
            model_source = model_info.get('source', 'unknown')
            
            # Debug: Log model information
            logger.info(f"🔍 Model Debug Info:")
            logger.info(f"   • Model name: {model_name}")
            logger.info(f"   • Model ID: {model_id}")
            logger.info(f"   • Model type: {model_type}")
            logger.info(f"   • Source: {model_source}")
            logger.info(f"   • Base model: {model_info.get('base_model', 'N/A')}")
            
            # Handle local files (Civitai checkpoints, safetensors, etc.)
            if model_source == 'local_file':
                local_path = model_info.get('path')
                if local_path:
                    from pathlib import Path
                    model_file = Path(local_path)
                    
                    if model_file.exists():
                        logger.info(f"🎯 Loading local checkpoint file: {local_path}")
                        
                        # Determine device and dtype
                        device = self.hardware_detector.get_optimal_device()
                        if self.hardware_detector.hardware_type == "apple_silicon":
                            torch_dtype = torch.float16
                        elif torch.cuda.is_available():
                            torch_dtype = torch.float16
                        else:
                            torch_dtype = torch.float32
                        
                        if progress_callback:
                            progress_callback(f"Loading local model file {model_name}...")
                        
                        # Load from single file - auto-detect pipeline type
                        try:
                            # First try AutoPipeline for automatic detection
                            logger.info(f"🚀 Attempting to load with AutoPipeline from: {local_path}")
                            pipeline = AutoPipelineForText2Image.from_single_file(
                                local_path,
                                torch_dtype=torch_dtype,
                                use_safetensors=local_path.endswith('.safetensors'),
                                safety_checker=None,  # Disable safety checker for uncensored models
                                requires_safety_checker=False
                            )
                            logger.info("✅ Local checkpoint loaded with AutoPipeline!")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ AutoPipeline failed: {e}")
                            
                            # Try SDXL pipeline if it's likely an XL model
                            if 'xl' in model_name.lower() or 'sdxl' in model_name.lower():
                                try:
                                    logger.info("🔄 Trying StableDiffusionXLPipeline...")
                                    pipeline = StableDiffusionXLPipeline.from_single_file(
                                        local_path,
                                        torch_dtype=torch_dtype,
                                        use_safetensors=local_path.endswith('.safetensors'),
                                        safety_checker=None,
                                        requires_safety_checker=False
                                    )
                                    logger.info("✅ Local checkpoint loaded with SDXL Pipeline!")
                                    
                                except Exception as e2:
                                    logger.warning(f"⚠️ SDXL Pipeline failed: {e2}")
                                    # Try standard pipeline as final fallback
                                    logger.info("🔄 Trying standard StableDiffusionPipeline...")
                                    pipeline = StableDiffusionPipeline.from_single_file(
                                        local_path,
                                        torch_dtype=torch_dtype,
                                        use_safetensors=local_path.endswith('.safetensors'),
                                        safety_checker=None,
                                        requires_safety_checker=False
                                    )
                                    logger.info("✅ Local checkpoint loaded with standard Pipeline!")
                            else:
                                # Try standard pipeline for non-XL models
                                logger.info("🔄 Trying standard StableDiffusionPipeline...")
                                pipeline = StableDiffusionPipeline.from_single_file(
                                    local_path,
                                    torch_dtype=torch_dtype,
                                    use_safetensors=local_path.endswith('.safetensors'),
                                    safety_checker=None,
                                    requires_safety_checker=False
                                )
                                logger.info("✅ Local checkpoint loaded with standard Pipeline!")
                        
                        # Move to device
                        logger.info(f"📱 Moving local checkpoint pipeline to device: {device}")
                        pipeline = pipeline.to(device)
                        
                        # Basic optimizations for local checkpoints
                        if hasattr(pipeline, 'enable_attention_slicing'):
                            pipeline.enable_attention_slicing()
                            logger.info("✅ Attention slicing enabled for local checkpoint")
                        
                        if hasattr(pipeline, 'enable_memory_efficient_attention') and torch.cuda.is_available():
                            try:
                                pipeline.enable_memory_efficient_attention()
                                logger.info("✅ Memory efficient attention enabled for local checkpoint")
                            except Exception:
                                pass  # Not all models support this
                        
                        if progress_callback:
                            progress_callback(f"Local checkpoint {model_name} loaded successfully!")
                        
                        logger.info(f"🎉 Successfully loaded local checkpoint: {model_name}")
                        return pipeline
                        
                    else:
                        logger.error(f"❌ Local model file not found: {local_path}")
                        return None
                else:
                    logger.error(f"❌ No path specified for local model: {model_name}")
                    return None
            
            # Continue with existing HuggingFace model loading logic...
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
            
            # Load pipeline based on model type - Check more specific types first
            if model_type == "flux_lora":
                # FLUX with LoRA (like Flux-NSFW-uncensored) - Handle this FIRST
                base_model = model_info.get('base_model', 'black-forest-labs/FLUX.1-dev')
                
                logger.info(f"🔄 Loading FLUX LoRA model: {model_id}")
                logger.info(f"🔧 Base model: {base_model}")
                
                if progress_callback:
                    progress_callback(f"Loading base model {base_model}...")
                
                # Get authentication token for gated models
                token = self._get_auth_token()
                
                logger.info("🚀 Loading base FLUX model...")
                try:
                    # Try with fp16 variant first
                    pipeline = FluxPipeline.from_pretrained(
                        base_model,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None,
                        token=token
                    )
                    logger.info("✅ Base FLUX model loaded successfully!")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load with fp16 variant: {e}")
                    
                    if "variant" in str(e).lower() and "fp16" in str(e):
                        logger.info("🔄 Retrying without fp16 variant...")
                        try:
                            # Try without variant
                            pipeline = FluxPipeline.from_pretrained(
                                base_model,
                                torch_dtype=torch_dtype,
                                use_safetensors=True,
                                token=token
                            )
                            logger.info("✅ Base FLUX model loaded without variant!")
                        except Exception as e2:
                            logger.error(f"❌ Failed even without variant: {e2}")
                            if "gated repo" in str(e2).lower() or "401" in str(e2):
                                logger.error(f"🚫 Base model {base_model} requires special access!")
                                logger.error("💡 Try using 'black-forest-labs/FLUX.1-schnell' instead")
                                return None
                            
                            logger.warning("🔄 Trying without authentication...")
                            try:
                                pipeline = FluxPipeline.from_pretrained(
                                    base_model,
                                    torch_dtype=torch_dtype,
                                    use_safetensors=True
                                )
                                logger.info("✅ Base FLUX model loaded without auth and variant!")
                            except Exception as e3:
                                logger.error(f"❌ All fallbacks failed: {e3}")
                                return None
                    else:
                        if "gated repo" in str(e).lower() or "401" in str(e):
                            logger.error(f"🚫 Base model {base_model} requires special access!")
                            logger.error("💡 Try using 'black-forest-labs/FLUX.1-schnell' instead")
                            return None
                        
                        logger.warning("🔄 Trying without authentication...")
                        try:
                            pipeline = FluxPipeline.from_pretrained(
                                base_model,
                                torch_dtype=torch_dtype,
                                use_safetensors=True,
                                variant="fp16" if torch_dtype == torch.float16 else None
                            )
                            logger.info("✅ Base FLUX model loaded without authentication!")
                        except Exception as e2:
                            logger.error(f"❌ Fallback failed: {e2}")
                            if "variant" in str(e2).lower():
                                try:
                                    pipeline = FluxPipeline.from_pretrained(
                                        base_model,
                                        torch_dtype=torch_dtype,
                                        use_safetensors=True
                                    )
                                    logger.info("✅ Base FLUX model loaded without auth and variant!")
                                except Exception as e3:
                                    logger.error(f"❌ All fallbacks failed: {e3}")
                                    return None
                            else:
                                logger.error(f"❌ Base model fallback failed: {e2}")
                                return None
                
                # Load LoRA weights
                lora_weight = model_info.get('lora_weight', 'lora.safetensors')
                
                logger.info(f"🔄 Loading LoRA weights: {lora_weight}")
                if progress_callback:
                    progress_callback(f"Loading LoRA weights...")
                
                try:
                    pipeline.load_lora_weights(
                        model_id,
                        weight_name=lora_weight,
                        adapter_name="uncensored",
                        token=token
                    )
                    logger.info("✅ LoRA weights loaded successfully!")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load LoRA with authentication: {e}")
                    try:
                        # Try without authentication
                        pipeline.load_lora_weights(
                            model_id,
                            weight_name=lora_weight,
                            adapter_name="uncensored"
                        )
                        logger.info("✅ LoRA weights loaded without authentication!")
                    except Exception as e2:
                        logger.error(f"❌ LoRA loading failed completely: {e2}")
                        logger.warning("🔄 Continuing with base model only...")
                        # Continue with base model only
                
            elif model_type == "flux" or "flux" in model_name.lower():
                # FLUX models (normal, not LoRA)
                token = self._get_auth_token()
                
                logger.info(f"🔄 Loading FLUX model: {model_id}")
                logger.info(f"🔑 Using token: {'Yes' if token else 'No'}")
                logger.info(f"🖥️  Device: {device}, Dtype: {torch_dtype}")
                
                try:
                    logger.info("🚀 Attempting to load FLUX pipeline with authentication...")
                    pipeline = FluxPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None,
                        token=token  # Updated syntax for authentication
                    )
                    logger.info("✅ FLUX pipeline loaded successfully!")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load FLUX model: {e}")
                    
                    if "variant" in str(e).lower() and "fp16" in str(e):
                        logger.info("🔄 Retrying without fp16 variant...")
                        try:
                            pipeline = FluxPipeline.from_pretrained(
                                model_id,
                                torch_dtype=torch_dtype,
                                use_safetensors=True,
                                token=token
                            )
                            logger.info("✅ FLUX pipeline loaded without variant!")
                        except Exception as e2:
                            logger.error(f"❌ Failed without variant: {e2}")
                            if "gated repo" in str(e2).lower() or "401" in str(e2):
                                logger.error(f"🚫 Model {model_id} requires special access!")
                                logger.error("💡 Solutions:")
                                logger.error("   1. Request access to the model on HuggingFace")
                                logger.error("   2. Use a different model")
                                logger.error("   3. Check your HF token permissions")
                                return None
                            
                            logger.warning("🔄 Trying without authentication...")
                            try:
                                pipeline = FluxPipeline.from_pretrained(
                                    model_id,
                                    torch_dtype=torch_dtype,
                                    use_safetensors=True
                                )
                                logger.info("✅ FLUX pipeline loaded without auth and variant!")
                            except Exception as e3:
                                logger.error(f"❌ All fallbacks failed: {e3}")
                                return None
                    elif "gated repo" in str(e).lower() or "401" in str(e):
                        logger.error(f"🚫 Model {model_id} requires special access!")
                        logger.error("💡 Solutions:")
                        logger.error("   1. Request access to the model on HuggingFace")
                        logger.error("   2. Use a different model")
                        logger.error("   3. Check your HF token permissions")
                        return None
                    else:
                        logger.warning("🔄 Trying without authentication as fallback...")
                        try:
                            pipeline = FluxPipeline.from_pretrained(
                                model_id,
                                torch_dtype=torch_dtype,
                                use_safetensors=True,
                                variant="fp16" if torch_dtype == torch.float16 else None
                            )
                            logger.info("✅ FLUX pipeline loaded without authentication!")
                        except Exception as e2:
                            logger.error(f"❌ Fallback failed: {e2}")
                            if "variant" in str(e2).lower():
                                try:
                                    pipeline = FluxPipeline.from_pretrained(
                                        model_id,
                                        torch_dtype=torch_dtype,
                                        use_safetensors=True
                                    )
                                    logger.info("✅ FLUX pipeline loaded without auth and variant!")
                                except Exception as e3:
                                    logger.error(f"❌ All FLUX fallbacks failed: {e3}")
                                    return None
                            else:
                                return None
                
            elif "xl" in model_name.lower() or model_type == "sdxl":
                # Stable Diffusion XL
                token = self._get_auth_token()
                
                try:
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None,
                        token=token
                    )
                except Exception as e:
                    logger.warning(f"Failed to load SDXL model with authentication: {e}")
                    # Try without authentication as fallback
                    pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        variant="fp16" if torch_dtype == torch.float16 else None
                    )
                
            else:
                # Standard Stable Diffusion or other models
                token = self._get_auth_token()
                
                try:
                    # Try AutoPipeline first
                    try:
                        pipeline = AutoPipelineForText2Image.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            use_safetensors=True,
                            variant="fp16" if torch_dtype == torch.float16 else None,
                            token=token
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load AutoPipeline with authentication: {e}")
                        # Try without authentication
                        pipeline = AutoPipelineForText2Image.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            use_safetensors=True,
                            variant="fp16" if torch_dtype == torch.float16 else None
                        )
                except:
                    # Fallback to standard pipeline
                    try:
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            use_safetensors=True,
                            variant="fp16" if torch_dtype == torch.float16 else None,
                            token=token
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load StableDiffusion pipeline with authentication: {e}")
                        # Try without authentication
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=torch_dtype,
                            use_safetensors=True,
                            variant="fp16" if torch_dtype == torch.float16 else None
                        )
            
            # Move to device
            logger.info(f"Moving pipeline to device: {device} - first time we are moving the pipeline to the device") 
            logger.info(f"might take a while, please wait...")
            pipeline = pipeline.to(device)
            logger.info(f"Pipeline moved to device: {device}")
            
            # Determine model type for optimization configuration
            is_flux_lora = model_type == "flux_lora"
            is_flux_model = model_type == "flux" or "flux" in model_name.lower()
            
            # Enable memory efficient attention if available
            if hasattr(pipeline, 'enable_attention_slicing'):
                logger.info("Enabling attention slicing")
                pipeline.enable_attention_slicing()
            
            if hasattr(pipeline, 'enable_memory_efficient_attention') and torch.cuda.is_available() and not is_flux_lora:
                try:
                    if is_flux_model:
                        logger.info("⚠️  Skipping memory efficient attention for FLUX model")
                        logger.info("🔧 Using standard memory management")
                    else:
                        logger.info("Enabling memory efficient attention")
                        pipeline.enable_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Memory efficient attention not available: {e}")
                    pass  # Some models don't support this
            elif is_flux_lora or is_flux_model:
                logger.info("🚫 Memory efficient attention disabled for FLUX models (compatibility)")
            
            # Enable xformers if available (but skip for FLUX+LoRA due to compatibility issues)
            
            if hasattr(pipeline, 'enable_xformers_memory_efficient_attention') and not is_flux_lora:
                try:
                    if is_flux_model:
                        logger.info("⚠️  Skipping xformers for FLUX model due to compatibility issues")
                        logger.info("🔧 Using standard attention mechanism instead")
                    else:
                        logger.info("Enabling xformers memory efficient attention")
                        pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(f"Xformers memory efficient attention not available: {e}")
                    pass  # xformers not available
            elif is_flux_lora:
                logger.info("🚫 Xformers disabled for FLUX+LoRA model (compatibility)")
                logger.info("🔧 Using standard attention mechanism for better stability")
            elif is_flux_model:
                logger.info("🚫 Xformers disabled for FLUX model (compatibility)")
                logger.info("🔧 Using standard attention mechanism for better stability")
            
            # Apple Silicon optimizations
            if self.hardware_detector.hardware_type == "apple_silicon":
                if hasattr(pipeline, 'enable_attention_slicing'):
                    logger.info("Enabling attention slicing on Apple Silicon")
                    pipeline.enable_attention_slicing()
            
            if progress_callback:
                progress_callback(f"Pipeline loaded successfully on {device}")
            else:
                logger.info(f"Pipeline loaded successfully on {device}")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading pipeline for {model_name}: {e}")
            if progress_callback:
                progress_callback(f"Error loading model: {e}")
            else:
                logger.error(f"Error loading model: {e}")
            return None
    
    def _create_placeholder_image(
        self, 
        width: int, 
        height: int, 
        prompt: str,
        model_name: str
    ) -> Image.Image:
        """Create a placeholder image for demonstration
        
        In a real implementation, this would be replaced with actual model inference
        """
        # Create a gradient image as placeholder
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a color based on prompt hash
        prompt_hash = hash(prompt) % 1000000
        r = (prompt_hash % 256)
        g = ((prompt_hash // 256) % 256)
        b = ((prompt_hash // 65536) % 256)
        
        # Create gradient
        for y in range(height):
            for x in range(width):
                # Gradient based on position
                gradient_r = int(r * (x / width))
                gradient_g = int(g * (y / height))
                gradient_b = int(b * ((x + y) / (width + height)))
                
                image_array[y, x] = [gradient_r, gradient_g, gradient_b]
        
        # Convert to PIL Image
        image = Image.fromarray(image_array)
        
        # Add some text (model name and truncated prompt)
        try:
            from PIL import ImageDraw, ImageFont
            
            draw = ImageDraw.Draw(image)
            
            # Try to use a default font
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Add model name
            draw.text((10, 10), f"Model: {model_name}", fill=(255, 255, 255), font=font)
            
            # Add truncated prompt
            truncated_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
            draw.text((10, 40), f"Prompt: {truncated_prompt}", fill=(255, 255, 255), font=font)
            
        except ImportError:
            # If PIL text rendering is not available, skip text
            pass
        
        return image
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available image models including downloaded models"""
        # Start with supported image models
        models = config.SUPPORTED_IMAGE_MODELS.copy()
        
        # Add downloaded models via model_manager
        try:
            from .models.model_manager import model_manager
            downloaded_models = model_manager._scan_downloaded_models()
            
            # Filter for image generation models
            for name, info in downloaded_models.items():
                # Include if it's for image generation or if type suggests image use
                content_type = info.get('content_type', 'image')
                model_type = info.get('type', 'unknown')
                
                if (content_type == 'image' or 
                    model_type in ['flux', 'stable_diffusion', 'lora', 'checkpoint'] or
                    any(keyword in name.lower() for keyword in ['diffusion', 'flux', 'xl', 'lora'])):
                    
                    # Add LoRA indicator to name if it's a LoRA
                    display_name = name
                    if model_type == 'lora':
                        display_name = f"{name} (LoRA)"
                    
                    models[display_name] = info
                    
        except Exception as e:
            logger.warning(f"Error loading downloaded image models: {e}")
        
        return models
    
    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """Search for image models"""
        return self._search_image_models(query)
    
    def get_recommended_models(self) -> List[str]:
        """Get recommended models for current hardware"""
        hardware_type = self.hardware_detector.hardware_type
        
        if hardware_type == "apple_silicon":
            return ["FLUX.1-schnell", "Stable Diffusion 2.1", "DreamShaper"]
        elif hardware_type == "nvidia_gpu":
            return ["Recomended.NSFW", "Flux-NSFW-uncensored", "FLUX.1-dev", "Stable Diffusion XL"]
        else:
            return ["Stable Diffusion 2.1", "DreamShaper"]
    
    def clear_pipeline_cache(self):
        """Clear the pipeline cache to free memory"""
        if self._pipeline_cache:
            logger.info(f"🗑️ Clearing pipeline cache ({len(self._pipeline_cache)} models)")
            
            # Free GPU memory if possible
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            self._pipeline_cache.clear()
            self._current_model = None
            self._current_pipeline = None
            logger.info("✅ Pipeline cache cleared")
        else:
            logger.info("📋 Pipeline cache is already empty")
    
    def get_cached_models(self) -> List[str]:
        """Get list of currently cached models"""
        return list(self._pipeline_cache.keys())
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state"""
        return {
            "cached_models": len(self._pipeline_cache),
            "cached_model_names": list(self._pipeline_cache.keys()),
            "current_model": self._current_model,
            "memory_usage": "Available" if self._pipeline_cache else "Empty"
        }
    
    def is_model_cached(self, model_name: str, model_info: Dict[str, Any]) -> bool:
        """Check if a specific model is cached"""
        cache_key = f"{model_name}_{model_info.get('model_id', model_name)}"
        return cache_key in self._pipeline_cache
    
    def _handle_lora_model(self, model_name: str, model_info: Dict[str, Any], progress_callback: Optional[callable] = None, preferred_base_model: str = None):
        """Handle LoRA model loading by finding a compatible base model"""
        try:
            if progress_callback:
                progress_callback(f"LoRA detected: {model_name}, finding base model...")
            
            # Get compatible base models
            from .models.model_manager import model_manager
            base_models = model_manager.get_base_models_for_lora(model_name)
            
            if not base_models:
                logger.error(f"No compatible base models found for LoRA: {model_name}")
                return None
            
            # Use the user-selected base model if it's compatible, otherwise use the first compatible one
            base_model_name = None
            if preferred_base_model and preferred_base_model in base_models:
                base_model_name = preferred_base_model
                logger.info(f"✅ Using user-selected base model: {preferred_base_model} for LoRA: {model_name}")
            else:
                # Fallback to first compatible base model
                base_model_name = base_models[0]
                logger.warning(f"⚠️ User-selected base model '{preferred_base_model}' not compatible with LoRA {model_name}")
                logger.info(f"📋 Using first compatible base model: {base_model_name}")
            
            if progress_callback:
                progress_callback(f"Using base model: {base_model_name} with LoRA: {model_name}")
            
            # Get base model info
            all_models = model_manager.get_available_models()
            base_model_info = all_models.get(base_model_name)
            
            if not base_model_info:
                logger.error(f"Base model info not found: {base_model_name}")
                return None
            
            # Load base pipeline
            base_pipeline = self._load_pipeline(base_model_name, base_model_info, progress_callback)
            
            if not base_pipeline:
                logger.error(f"Failed to load base pipeline: {base_model_name}")
                return None
            
            # Apply LoRA to the pipeline
            lora_path = model_info.get('path')
            if lora_path and Path(lora_path).exists():
                try:
                    if progress_callback:
                        progress_callback(f"Applying LoRA: {model_name}...")
                    
                    # Load LoRA weights
                    if hasattr(base_pipeline, 'load_lora_weights'):
                        base_pipeline.load_lora_weights(lora_path)
                        logger.info(f"✅ LoRA applied successfully: {model_name}")
                        
                        if progress_callback:
                            progress_callback(f"LoRA {model_name} applied to {base_model_name}")
                        
                        return base_pipeline
                    else:
                        logger.warning(f"Base model {base_model_name} doesn't support LoRA loading")
                        return base_pipeline
                        
                except Exception as e:
                    logger.warning(f"Failed to apply LoRA {model_name}: {e}")
                    logger.info(f"Using base model {base_model_name} without LoRA")
                    return base_pipeline
            else:
                logger.error(f"LoRA file not found: {lora_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error handling LoRA model {model_name}: {e}")
            return None
    
    def generate_with_separate_models(
        self,
        prompt: str,
        base_model_name: str = None,
        lora_model_name: str = None,
        style: str = None,
        resolution: str = None,
        output_format: str = None,
        quality: str = None,
        reference_images: Optional[List[str]] = None,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 28,
        seed: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Optional[str]:
        """Generate an image with separate base model and LoRA selection
        
        Args:
            prompt: Text prompt for image generation
            base_model_name: Name of the base model to use
            lora_model_name: Name of the LoRA model to use (or "none")
            style: Image style
            resolution: Image resolution
            output_format: Output image format
            quality: Quality preset
            reference_images: List of reference image paths
            negative_prompt: Negative prompt
            guidance_scale: Guidance scale for generation
            num_inference_steps: Number of inference steps
            seed: Random seed for reproducibility
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to generated image or None if failed
        """
        try:
            # Debug logging
            logger.info(f"🔧 generate_with_separate_models called:")
            logger.info(f"   base_model_name: '{base_model_name}'")
            logger.info(f"   lora_model_name: '{lora_model_name}'")
            
            # Validate inputs
            if not base_model_name:
                logger.error("❌ No base model specified")
                return None
            
            # Determine which model to use for generation
            if lora_model_name and lora_model_name != "none":
                # Use LoRA model as primary
                primary_model = lora_model_name
                logger.info(f"🎨 Using LoRA model: {lora_model_name} with base: {base_model_name}")
                
                # TODO: In the future, we could implement actual LoRA combination here
                # For now, we'll use the LoRA model if it exists, otherwise fall back to base
                available_models = self.get_available_models()
                if primary_model not in available_models:
                    logger.warning(f"⚠️ LoRA model '{primary_model}' not found, using base model: {base_model_name}")
                    primary_model = base_model_name
            else:
                # Use base model only
                primary_model = base_model_name
                logger.info(f"🖼️ Using base model only: {base_model_name}")
            
            logger.info(f"🎯 Final model selection: '{primary_model}'")
            
            # Call the original generate method with the selected model
            return self.generate(
                prompt=prompt,
                model_name=primary_model,
                style=style,
                resolution=resolution,
                output_format=output_format,
                quality=quality,
                reference_images=reference_images,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                progress_callback=progress_callback,
                preferred_base_model=base_model_name
            )
            
        except Exception as e:
            logger.error(f"❌ Error in generate_with_separate_models: {e}")
            return None


# Global image generator instance
image_generator = ImageGenerator() 