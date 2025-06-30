"""
Hugging Face client for model discovery and downloading
"""
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
import json
from huggingface_hub import HfApi, hf_hub_download, list_models, model_info
from huggingface_hub.utils import HfHubHTTPError
import time

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a Hugging Face model"""
    model_id: str
    model_name: str
    description: str
    downloads: int
    likes: int
    tags: List[str]
    pipeline_tag: Optional[str]
    library_name: Optional[str]
    created_at: str
    last_modified: str
    model_type: str
    memory_requirement: str
    supports_text_prompt: bool
    supports_image_input: bool
    max_duration: int


class HuggingFaceClient:
    """Client for interacting with Hugging Face Hub"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Hugging Face client
        
        Args:
            api_key: Optional API key. If not provided, will try to get from environment
        """
        # Get API key from parameter, environment variable, or HF cache
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN") or "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"
        
        # Initialize Hugging Face API
        self.hf_api = HfApi(token=self.api_key)
        
        # Video generation model tags to search for
        self.video_tags = [
            "text-to-video",
            "image-to-video", 
            "video-generation",
            "diffusion",
            "stable-video-diffusion",
            "animatediff",
            "video-synthesis",
            "video-diffusion",
            "text2video",
            "img2vid"
        ]
        
        # Cache for search results
        self._search_cache = {}
        self._cache_ttl = 3600  # 1 hour
        
        logger.info(f"Initialized Hugging Face client{'with API key' if self.api_key else 'without API key'}")
    
    def search_video_models(
        self, 
        query: str = "", 
        limit: int = 20,
        sort: str = "downloads",
        direction: int = -1
    ) -> List[ModelInfo]:
        """Search for video generation models on Hugging Face
        
        Args:
            query: Search query string
            limit: Maximum number of results
            sort: Sort by ('downloads', 'created_at', 'last_modified')
            direction: Sort direction (1 for ascending, -1 for descending)
            
        Returns:
            List of ModelInfo objects
        """
        cache_key = f"{query}_{limit}_{sort}_{direction}"
        
        # Check cache first
        if cache_key in self._search_cache:
            cached_time, cached_results = self._search_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                logger.debug(f"Using cached results for query: {query}")
                return cached_results
        
        try:
            logger.info(f"Searching for video models with query: '{query}'")
            
            # Search for models with video generation tags
            models = list_models(
                search=query,
                tags=self.video_tags,
                sort=sort,
                direction=direction,
                limit=limit * 2,  # Get more to filter later
                token=self.api_key
            )
            
            results = []
            for model in models:
                try:
                    model_info_obj = self._parse_model_info(model)
                    if model_info_obj and self._is_video_generation_model(model):
                        results.append(model_info_obj)
                        
                    if len(results) >= limit:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error parsing model {model.modelId}: {e}")
                    continue
            
            # Cache results
            self._search_cache[cache_key] = (time.time(), results)
            
            logger.info(f"Found {len(results)} video generation models")
            return results
            
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []
    
    def get_featured_models(self) -> List[ModelInfo]:
        """Get a curated list of featured video generation models"""
        featured_model_ids = [
            "stabilityai/stable-video-diffusion-img2vid-xt",
            "guoyww/animatediff-motion-adapter-v1-5-2", 
            "damo-vilab/text-to-video-ms-1.7b",
            "cerspense/zeroscope_v2_576w",
            "ali-vilab/i2vgen-xl",
            "camenduru/potat1"
        ]
        
        results = []
        for model_id in featured_model_ids:
            try:
                model_info_obj = self.get_model_info(model_id)
                if model_info_obj:
                    results.append(model_info_obj)
            except Exception as e:
                logger.warning(f"Could not get info for featured model {model_id}: {e}")
                continue
        
        return results
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model
        
        Args:
            model_id: Hugging Face model ID
            
        Returns:
            ModelInfo object or None if not found
        """
        try:
            logger.debug(f"Getting info for model: {model_id}")
            
            info = model_info(model_id, token=self.api_key)
            return self._parse_model_info(info)
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Model not found: {model_id}")
            else:
                logger.error(f"HTTP error getting model info for {model_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {e}")
            return None
    
    def download_model(
        self, 
        model_id: str, 
        local_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False
    ) -> str:
        """Download a model from Hugging Face
        
        Args:
            model_id: Hugging Face model ID
            local_dir: Local directory to save the model
            cache_dir: Cache directory (uses HF default if not specified)
            force_download: Force re-download even if cached
            
        Returns:
            Path to downloaded model
        """
        try:
            logger.info(f"Downloading model: {model_id}")
            
            # For now, just ensure the model is in cache
            # Full download will be handled by the actual model loading
            from huggingface_hub import snapshot_download
            
            path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_dir=local_dir,
                token=self.api_key,
                force_download=force_download,
                local_dir_use_symlinks=False if local_dir else True
            )
            
            logger.info(f"Model downloaded to: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            raise
    
    def _parse_model_info(self, model) -> Optional[ModelInfo]:
        """Parse model information from HF API response"""
        try:
            # Extract basic info
            model_id = model.modelId if hasattr(model, 'modelId') else model.id
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            
            # Get tags and pipeline info
            tags = getattr(model, 'tags', []) or []
            pipeline_tag = getattr(model, 'pipeline_tag', None)
            library_name = getattr(model, 'library_name', None)
            
            # Get stats
            downloads = getattr(model, 'downloads', 0) or 0
            likes = getattr(model, 'likes', 0) or 0
            
            # Get dates
            created_at = getattr(model, 'created_at', '').isoformat() if hasattr(getattr(model, 'created_at', None), 'isoformat') else str(getattr(model, 'created_at', ''))
            last_modified = getattr(model, 'last_modified', '').isoformat() if hasattr(getattr(model, 'last_modified', None), 'isoformat') else str(getattr(model, 'last_modified', ''))
            
            # Determine model characteristics
            model_type = self._determine_model_type(tags, pipeline_tag)
            memory_requirement = self._estimate_memory_requirement(model_id, tags)
            supports_text_prompt = self._supports_text_prompt(tags, model_id)
            supports_image_input = self._supports_image_input(tags, model_id)
            max_duration = self._estimate_max_duration(model_id, tags)
            
            # Generate description
            description = self._generate_description(model_id, tags, pipeline_tag)
            
            return ModelInfo(
                model_id=model_id,
                model_name=model_name,
                description=description,
                downloads=downloads,
                likes=likes,
                tags=tags,
                pipeline_tag=pipeline_tag,
                library_name=library_name,
                created_at=created_at,
                last_modified=last_modified,
                model_type=model_type,
                memory_requirement=memory_requirement,
                supports_text_prompt=supports_text_prompt,
                supports_image_input=supports_image_input,
                max_duration=max_duration
            )
            
        except Exception as e:
            logger.error(f"Error parsing model info: {e}")
            return None
    
    def _is_video_generation_model(self, model) -> bool:
        """Check if model is suitable for video generation"""
        tags = getattr(model, 'tags', []) or []
        pipeline_tag = getattr(model, 'pipeline_tag', None)
        model_id = model.modelId if hasattr(model, 'modelId') else model.id
        
        # Check tags
        video_keywords = [
            'video', 'text-to-video', 'image-to-video', 'animatediff', 
            'stable-video-diffusion', 'video-generation', 'video-synthesis'
        ]
        
        has_video_tag = any(keyword in ' '.join(tags).lower() for keyword in video_keywords)
        has_video_pipeline = pipeline_tag in ['text-to-video', 'image-to-video']
        has_video_name = any(keyword in model_id.lower() for keyword in video_keywords)
        
        return has_video_tag or has_video_pipeline or has_video_name
    
    def _determine_model_type(self, tags: List[str], pipeline_tag: Optional[str]) -> str:
        """Determine the type of model based on tags and pipeline"""
        if 'diffusion' in ' '.join(tags).lower() or 'diffusers' in ' '.join(tags).lower():
            return 'diffusion'
        elif 'transformer' in ' '.join(tags).lower():
            return 'transformer'
        elif pipeline_tag:
            return pipeline_tag
        else:
            return 'unknown'
    
    def _estimate_memory_requirement(self, model_id: str, tags: List[str]) -> str:
        """Estimate memory requirements based on model info"""
        model_id_lower = model_id.lower()
        
        # High memory models
        if any(keyword in model_id_lower for keyword in ['xl', 'large', 'videocrafter']):
            return '16GB'
        # Medium-high memory models
        elif any(keyword in model_id_lower for keyword in ['stable-video-diffusion', 'i2vgen']):
            return '12GB'
        # Medium memory models
        elif any(keyword in model_id_lower for keyword in ['animatediff', 'lavie']):
            return '8GB'
        # Lower memory models
        else:
            return '6GB'
    
    def _supports_text_prompt(self, tags: List[str], model_id: str) -> bool:
        """Check if model supports text prompts"""
        text_keywords = ['text-to-video', 'text2video', 'prompt']
        tags_str = ' '.join(tags).lower()
        model_id_lower = model_id.lower()
        
        return any(keyword in tags_str or keyword in model_id_lower for keyword in text_keywords)
    
    def _supports_image_input(self, tags: List[str], model_id: str) -> bool:
        """Check if model supports image input"""
        image_keywords = ['image-to-video', 'img2vid', 'i2v', 'image-conditional']
        tags_str = ' '.join(tags).lower()
        model_id_lower = model_id.lower()
        
        return any(keyword in tags_str or keyword in model_id_lower for keyword in image_keywords)
    
    def _estimate_max_duration(self, model_id: str, tags: List[str]) -> int:
        """Estimate maximum video duration in seconds"""
        model_id_lower = model_id.lower()
        
        # Long video models
        if 'lavie' in model_id_lower:
            return 180
        elif 'videocrafter' in model_id_lower:
            return 120
        elif 'stable-video-diffusion' in model_id_lower:
            return 25  # Actually 25 frames, not seconds
        else:
            return 30  # Default
    
    def _generate_description(self, model_id: str, tags: List[str], pipeline_tag: Optional[str]) -> str:
        """Generate a description for the model"""
        model_name = model_id.split('/')[-1]
        
        # Special cases for known models
        descriptions = {
            'stable-video-diffusion': 'High-quality video generation from images using Stable Video Diffusion',
            'animatediff': 'Animation-focused video generation with motion control',
            'text-to-video-ms': 'Text-to-video generation model from Alibaba DAMO',
            'videocrafter': 'High-resolution video synthesis with excellent quality',
            'lavie': 'Long video generation with temporal consistency',
            'i2vgen-xl': 'Image-to-video generation with fine control',
            'zeroscope': 'Efficient text-to-video generation model'
        }
        
        for key, desc in descriptions.items():
            if key in model_id.lower():
                return desc
        
        # Generate generic description
        if pipeline_tag == 'text-to-video':
            return f"Text-to-video generation model: {model_name}"
        elif pipeline_tag == 'image-to-video':
            return f"Image-to-video generation model: {model_name}"
        elif 'diffusion' in ' '.join(tags).lower():
            return f"Diffusion-based video generation model: {model_name}"
        else:
            return f"Video generation model: {model_name}"
    
    def get_popular_models(self, limit: int = 10) -> List[ModelInfo]:
        """Get popular video generation models"""
        return self.search_video_models(
            query="",
            limit=limit,
            sort="downloads",
            direction=-1
        )
    
    def get_recent_models(self, limit: int = 10) -> List[ModelInfo]:
        """Get recently updated video generation models"""
        return self.search_video_models(
            query="",
            limit=limit,
            sort="last_modified",
            direction=-1
        )
    
    def validate_api_key(self) -> bool:
        """Validate if the API key is working"""
        if not self.api_key:
            return False
        
        try:
            # Try to get user info
            self.hf_api.whoami()
            return True
        except Exception:
            return False


# Global client instance
hf_client = HuggingFaceClient() 