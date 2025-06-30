"""
Model manager for handling both default and on-demand models
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from ..config import config
from .huggingface_client import HuggingFaceClient, ModelInfo
from .civitai_client import CivitaiClient
from ..utils.hardware import hardware_detector

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages video generation models from default list, Hugging Face, and Civitai"""
    
    def __init__(
        self, 
        hf_client: Optional[HuggingFaceClient] = None,
        civitai_client: Optional[CivitaiClient] = None
    ):
        """Initialize model manager
        
        Args:
            hf_client: Optional Hugging Face client instance
            civitai_client: Optional Civitai client instance
        """
        self.hf_client = hf_client or HuggingFaceClient()
        self.civitai_client = civitai_client or CivitaiClient()
        self.default_models = config.SUPPORTED_MODELS.copy()
        self.cached_models = {}
        self.downloaded_models = set()
        self.lock = threading.Lock()
        
        # Model cache file
        self.cache_file = Path(config.MODEL_CACHE_DIR) / "model_cache.json"
        self.load_cache()
        
        logger.info("Model manager initialized with Hugging Face and Civitai support")
    
    def get_available_models(self, include_online: bool = True) -> Dict[str, Dict[str, Any]]:
        """Get all available models (default + online)
        
        Args:
            include_online: Whether to include online models from Hugging Face
            
        Returns:
            Dictionary of model names to model information
        """
        models = self.default_models.copy()
        
        if include_online:
            # Add featured models from Hugging Face
            try:
                featured_models = self.hf_client.get_featured_models()
                for model_info in featured_models:
                    # Convert HF ModelInfo to our format
                    models[model_info.model_name] = self._convert_hf_model_info(model_info)
                    
            except Exception as e:
                logger.warning(f"Could not fetch featured models: {e}")
        
        return models
    
    def search_models(
        self, 
        query: str, 
        limit: int = 20,
        include_default: bool = True,
        platforms: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for models by query across multiple platforms
        
        Args:
            query: Search query
            limit: Maximum number of results
            include_default: Whether to include default models in search
            platforms: List of platforms to search ("huggingface", "civitai", "default")
            
        Returns:
            List of model information dictionaries
        """
        if platforms is None:
            platforms = ["default", "huggingface", "civitai"]
        
        results = []
        
        # Search in default models
        if include_default and "default" in platforms:
            query_lower = query.lower()
            for model_name, model_info in self.default_models.items():
                if (query_lower in model_name.lower() or 
                    query_lower in model_info.get('description', '').lower()):
                    results.append({
                        'name': model_name,
                        'source': 'default',
                        **model_info
                    })
        
        # Search in Hugging Face
        if "huggingface" in platforms:
            try:
                hf_models = self.hf_client.search_video_models(query, limit=limit//2)
                for model_info in hf_models:
                    results.append({
                        'name': model_info.model_name,
                        'source': 'huggingface',
                        **self._convert_hf_model_info(model_info)
                    })
                    
            except Exception as e:
                logger.warning(f"Error searching Hugging Face models: {e}")
        
        # Search in Civitai
        if "civitai" in platforms:
            try:
                civitai_models = self.civitai_client.search_models(query, limit=limit//2)
                for model_info in civitai_models:
                    results.append({
                        'name': model_info.get('name', f"civitai_{model_info.get('id')}"),
                        'source': 'civitai',
                        **self._convert_civitai_model_info(model_info)
                    })
                    
            except Exception as e:
                logger.warning(f"Error searching Civitai models: {e}")
        
        return results[:limit]
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        # Check default models first
        if model_name in self.default_models:
            return {
                'name': model_name,
                'source': 'default',
                **self.default_models[model_name]
            }
        
        # Check cache
        if model_name in self.cached_models:
            return self.cached_models[model_name]
        
        # Try to get from Hugging Face by model ID
        try:
            hf_model = self.hf_client.get_model_info(model_name)
            if hf_model:
                model_info = {
                    'name': hf_model.model_name,
                    'source': 'huggingface',
                    **self._convert_hf_model_info(hf_model)
                }
                
                # Cache the result
                with self.lock:
                    self.cached_models[model_name] = model_info
                    self.save_cache()
                
                return model_info
                
        except Exception as e:
            logger.warning(f"Error getting model info for {model_name}: {e}")
        
        return None
    
    def download_model(
        self, 
        model_name: str, 
        model_id: Optional[str] = None,
        force: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Optional[str]:
        """Download a model from Hugging Face or Civitai
        
        Args:
            model_name: Name of the model
            model_id: Model ID (HF model ID or civitai:ID format)
            force: Force re-download even if already downloaded
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to downloaded model or None if failed
        """
        # Check if it's a default model (already "available")
        if model_name in self.default_models:
            logger.info(f"Model {model_name} is a default model, no download needed")
            return "default"
        
        # Use model_id if provided, otherwise use model_name
        download_id = model_id or model_name
        
        # Check if already downloaded
        if not force and download_id in self.downloaded_models:
            logger.info(f"Model {download_id} already downloaded")
            return "cached"
        
        try:
            if progress_callback:
                progress_callback(f"Starting download of {download_id}...")
            
            # Determine source and download accordingly
            if download_id.startswith("civitai:"):
                # Download from Civitai
                civitai_id = download_id.replace("civitai:", "")
                path = self.civitai_client.download_model(
                    model_id=civitai_id,
                    output_dir=config.MODEL_CACHE_DIR,
                    progress_callback=progress_callback
                )
            else:
                # Download from Hugging Face
                path = self.hf_client.download_model(
                    model_id=download_id,
                    cache_dir=config.MODEL_CACHE_DIR
                )
            
            if path:
                # Mark as downloaded (both by name and ID)
                with self.lock:
                    self.downloaded_models.add(download_id)
                    if model_name != download_id:
                        self.downloaded_models.add(model_name)
                    self.save_cache()
                
                if progress_callback:
                    progress_callback(f"Successfully downloaded {download_id}")
                
                logger.info(f"Successfully downloaded model {download_id} to {path}")
                return path
            else:
                error_msg = f"Download failed for {download_id}"
                logger.error(error_msg)
                if progress_callback:
                    progress_callback(error_msg)
                return None
            
        except Exception as e:
            error_msg = f"Failed to download model {download_id}: {e}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            return None
    
    def get_recommended_models(self, hardware_type: Optional[str] = None) -> List[str]:
        """Get recommended models based on hardware
        
        Args:
            hardware_type: Hardware type (auto-detected if not provided)
            
        Returns:
            List of recommended model names
        """
        if hardware_type is None:
            hardware_type = hardware_detector.hardware_type
        
        # Get hardware-specific recommendations
        hardware_config = config.HARDWARE_CONFIGS.get(hardware_type, {})
        recommended = hardware_config.get('recommended_models', [])
        
        # If no specific recommendations, provide general ones
        if not recommended:
            if hardware_type == "apple_silicon":
                recommended = ["AnimateDiff", "Text2Video-Zero", "I2VGen-XL"]
            elif hardware_type == "nvidia_gpu":
                recommended = ["VideoCrafter", "Stable Video Diffusion", "LaVie", "AnimateDiff"]
            else:
                recommended = ["Text2Video-Zero", "AnimateDiff"]
        
        return recommended
    
    def get_popular_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get popular models from Hugging Face
        
        Args:
            limit: Maximum number of models to return
            
        Returns:
            List of popular model information
        """
        try:
            hf_models = self.hf_client.get_popular_models(limit=limit)
            return [
                {
                    'name': model.model_name,
                    'source': 'huggingface',
                    **self._convert_hf_model_info(model)
                }
                for model in hf_models
            ]
        except Exception as e:
            logger.warning(f"Error getting popular models: {e}")
            return []
    
    def get_recent_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently updated models from Hugging Face
        
        Args:
            limit: Maximum number of models to return
            
        Returns:
            List of recent model information
        """
        try:
            hf_models = self.hf_client.get_recent_models(limit=limit)
            return [
                {
                    'name': model.model_name,
                    'source': 'huggingface',
                    **self._convert_hf_model_info(model)
                }
                for model in hf_models
            ]
        except Exception as e:
            logger.warning(f"Error getting recent models: {e}")
            return []
    
    def _convert_hf_model_info(self, hf_model: ModelInfo) -> Dict[str, Any]:
        """Convert Hugging Face ModelInfo to our format
        
        Args:
            hf_model: Hugging Face ModelInfo object
            
        Returns:
            Dictionary in our model format
        """
        return {
            'model_id': hf_model.model_id,
            'type': hf_model.model_type,
            'description': hf_model.description,
            'memory_requirement': hf_model.memory_requirement,
            'supports_text_prompt': hf_model.supports_text_prompt,
            'supports_image_input': hf_model.supports_image_input,
            'max_duration': hf_model.max_duration,
            'downloads': hf_model.downloads,
            'likes': hf_model.likes,
            'tags': hf_model.tags,
            'created_at': hf_model.created_at,
            'last_modified': hf_model.last_modified,
        }
    
    def _convert_civitai_model_info(self, civitai_model: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Civitai model info to our format
        
        Args:
            civitai_model: Civitai model information dictionary
            
        Returns:
            Dictionary in our model format
        """
        model_type = civitai_model.get('type', '').lower()
        
        # Determine if it's suitable for video or image generation
        is_video_model = model_type in ['checkpoint'] and 'video' in civitai_model.get('name', '').lower()
        is_image_model = model_type in ['checkpoint', 'lora', 'textualinversion', 'hypernetwork']
        
        return {
            'model_id': f"civitai:{civitai_model.get('id')}",
            'type': 'video_generation' if is_video_model else 'image_generation',
            'description': civitai_model.get('description', 'No description available'),
            'memory_requirement': civitai_model.get('memory_requirement', '8GB'),
            'supports_text_prompt': True,
            'supports_image_input': model_type == 'lora',
            'max_duration': 30 if is_video_model else None,
            'downloads': civitai_model.get('downloads', 0),
            'likes': civitai_model.get('likes', 0),
            'rating': civitai_model.get('rating', 0),
            'tags': civitai_model.get('tags', []),
            'creator': civitai_model.get('creator', 'Unknown'),
            'created_at': civitai_model.get('created_at'),
            'last_modified': civitai_model.get('updated_at'),
            'content_type': civitai_model.get('content_type', 'image'),
            'nsfw': civitai_model.get('nsfw', False),
            'civitai_id': civitai_model.get('id'),
            'civitai_type': model_type,
            'url': civitai_model.get('url'),
            'images': civitai_model.get('images', [])[:3]  # Limit to 3 preview images
        }
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available (default or downloaded)
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is available, False otherwise
        """
        # Check default models
        if model_name in self.default_models:
            return True
        
        # Check downloaded models by name
        if model_name in self.downloaded_models:
            return True
        
        # Check if it's a model ID that's been downloaded
        model_info = self.get_model_info(model_name)
        if model_info:
            model_id = model_info.get('model_id', model_name)
            if model_id in self.downloaded_models:
                return True
        
        return False
    
    def get_model_requirements(self, model_name: str) -> Dict[str, Any]:
        """Get hardware requirements for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with hardware requirements
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            return {}
        
        memory_str = model_info.get('memory_requirement', '8GB')
        memory_gb = int(memory_str.replace('GB', '')) if memory_str.replace('GB', '').isdigit() else 8
        
        return {
            'memory_gb': memory_gb,
            'supports_text_input': model_info.get('supports_text_prompt', True),
            'supports_image_input': model_info.get('supports_image_input', False),
            'max_duration': model_info.get('max_duration', 30),
            'model_type': model_info.get('type', 'unknown')
        }
    
    def load_cache(self):
        """Load cached model information from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self.cached_models = cache_data.get('cached_models', {})
                    self.downloaded_models = set(cache_data.get('downloaded_models', []))
                logger.debug(f"Loaded cache with {len(self.cached_models)} models")
        except Exception as e:
            logger.warning(f"Could not load model cache: {e}")
            self.cached_models = {}
            self.downloaded_models = set()
    
    def save_cache(self):
        """Save cached model information to disk"""
        try:
            os.makedirs(self.cache_file.parent, exist_ok=True)
            cache_data = {
                'cached_models': self.cached_models,
                'downloaded_models': list(self.downloaded_models)
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug("Saved model cache")
        except Exception as e:
            logger.warning(f"Could not save model cache: {e}")
    
    def clear_cache(self):
        """Clear the model cache"""
        with self.lock:
            self.cached_models.clear()
            self.downloaded_models.clear()
            if self.cache_file.exists():
                self.cache_file.unlink()
        logger.info("Cleared model cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the model cache
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cached_models': len(self.cached_models),
            'downloaded_models': len(self.downloaded_models),
            'cache_file_exists': self.cache_file.exists(),
            'cache_file_size': self.cache_file.stat().st_size if self.cache_file.exists() else 0,
        }


# Global model manager instance
model_manager = ModelManager() 