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
        """Get all available models (default + downloaded + online)
        
        Args:
            include_online: Whether to include online models from Hugging Face
            
        Returns:
            Dictionary of model names to model information
        """
        models = self.default_models.copy()
        
        # Add downloaded models
        downloaded_models = self._scan_downloaded_models()
        models.update(downloaded_models)
        
        if include_online:
            # Add featured models from Hugging Face
            try:
                featured_models = self.hf_client.get_featured_models()
                for model_info in featured_models:
                    # Don't overwrite downloaded models
                    if model_info.model_name not in models:
                        # Convert HF ModelInfo to our format
                        models[model_info.model_name] = self._convert_hf_model_info(model_info)
                    
            except Exception as e:
                logger.warning(f"Could not fetch featured models: {e}")
        
        return models
    
    def _scan_downloaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Scan the models directory for downloaded models"""
        downloaded_models = {}
        models_dir = Path(config.MODEL_CACHE_DIR)
        
        if not models_dir.exists():
            return downloaded_models
        
        try:
            # Scan for HuggingFace repository directories
            for item in models_dir.iterdir():
                if item.is_dir() and item.name.startswith("models--"):
                    model_info = self._analyze_hf_model_dir(item)
                    if model_info:
                        downloaded_models[model_info['name']] = model_info
            
            # Scan for individual model files (LoRAs, checkpoints, etc.)
            for item in models_dir.iterdir():
                if item.is_file() and item.suffix in ['.safetensors', '.ckpt', '.pt', '.pth']:
                    model_info = self._analyze_model_file(item)
                    if model_info:
                        downloaded_models[model_info['name']] = model_info
            
            logger.info(f"Found {len(downloaded_models)} downloaded models")
            
        except Exception as e:
            logger.warning(f"Error scanning downloaded models: {e}")
        
        return downloaded_models
    
    def _analyze_hf_model_dir(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """Analyze a HuggingFace model directory"""
        try:
            # Parse model name from directory (models--author--model-name)
            dir_parts = model_dir.name.split("--")
            if len(dir_parts) >= 3:
                author = dir_parts[1]
                model_name = dir_parts[2]
                full_name = f"{author}/{model_name}"
                
                # Check for snapshots directory
                snapshots_dir = model_dir / "snapshots"
                if snapshots_dir.exists():
                    # Find the latest snapshot
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                        
                        # Determine model type based on contents
                        model_type = self._determine_model_type_from_files(latest_snapshot)
                        
                        return {
                            'name': model_name,
                            'full_name': full_name,
                            'source': 'huggingface_downloaded',
                            'type': model_type,
                            'description': f'Downloaded HuggingFace model: {full_name}',
                            'path': str(model_dir),
                            'model_id': full_name,
                            'memory_requirement': self._estimate_memory_requirement(latest_snapshot),
                            'downloaded': True
                        }
        except Exception as e:
            logger.warning(f"Error analyzing HF model dir {model_dir}: {e}")
        
        return None
    
    def _analyze_model_file(self, model_file: Path) -> Optional[Dict[str, Any]]:
        """Analyze an individual model file"""
        try:
            file_size = model_file.stat().st_size
            
            # Determine model type based on filename and size
            filename = model_file.stem
            
            # Check if it's likely a LoRA (smaller files, usually under 500MB)
            is_likely_lora = file_size < 500 * 1024 * 1024  # 500MB threshold
            
            model_type = "lora" if is_likely_lora else "checkpoint"
            
            # Estimate memory requirement based on file size
            memory_gb = max(1, int(file_size / (1024**3)) + 2)  # File size + 2GB overhead
            memory_requirement = f"{memory_gb}GB"
            
            # Determine if it's for image or video generation based on filename
            content_type = "image"  # Most single files are image models
            if any(keyword in filename.lower() for keyword in ["video", "animate", "motion"]):
                content_type = "video"
            
            return {
                'name': filename,
                'source': 'local_file',
                'type': model_type,
                'content_type': content_type,
                'description': f'Local {model_type} file: {filename}',
                'path': str(model_file),
                'file_size': file_size,
                'memory_requirement': memory_requirement,
                'downloaded': True,
                'requires_base_model': is_likely_lora
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing model file {model_file}: {e}")
        
        return None
    
    def _determine_model_type_from_files(self, model_dir: Path) -> str:
        """Determine model type from files in the model directory"""
        try:
            files = list(model_dir.glob("*.json")) + list(model_dir.glob("*.py"))
            file_names = [f.name.lower() for f in files]
            
            # Check for specific model indicators
            if any("flux" in name for name in file_names):
                return "flux"
            elif "model_index.json" in file_names or "diffusers" in str(model_dir):
                if any("video" in name or "animate" in name for name in file_names):
                    return "video_diffusion"
                else:
                    return "stable_diffusion"
            elif any("config.json" in name for name in file_names):
                return "transformers_model"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"
    
    def _estimate_memory_requirement(self, model_dir: Path) -> str:
        """Estimate memory requirement from model directory size"""
        try:
            total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            # Rough estimate: model size + 50% overhead
            estimated_gb = max(2, int(total_size * 1.5 / (1024**3)))
            return f"{estimated_gb}GB"
        except Exception:
            return "Unknown"
    
    def get_available_loras(self) -> Dict[str, Dict[str, Any]]:
        """Get available LoRA models specifically"""
        all_models = self._scan_downloaded_models()
        loras = {name: info for name, info in all_models.items() 
                if info.get('type') == 'lora'}
        return loras
    
    def get_base_models_for_lora(self, lora_name: str) -> List[str]:
        """Get compatible base models for a LoRA"""
        # This is a simplified implementation
        # In practice, you'd want to check LoRA metadata for compatibility
        all_models = self.get_available_models()
        
        # Return models that are not LoRAs and are for image generation
        base_models = []
        for name, info in all_models.items():
            if (info.get('type') != 'lora' and 
                info.get('content_type', 'image') == 'image' and
                'flux' in name.lower()):  # Simplified compatibility check
                base_models.append(name)
        
        return base_models
    
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
        progress_callback: Optional[callable] = None,
        model_type_hint: Optional[str] = None
    ) -> Optional[str]:
        """Download a model from Hugging Face or Civitai
        
        Args:
            model_name: Name of the model
            model_id: Model ID (HF model ID or civitai:ID format)
            force: Force re-download even if already downloaded
            progress_callback: Optional callback for progress updates
            model_type_hint: Optional hint about intended use (video/image)
            
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
                if model_type_hint:
                    progress_callback(f"Model intended for {model_type_hint} generation")
            
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