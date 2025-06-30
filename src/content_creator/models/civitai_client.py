"""
Civitai client for Content Creator
"""
import os
import logging
import requests
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class CivitaiClient:
    """Client for interacting with Civitai API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Civitai client
        
        Args:
            api_key: Civitai API token (optional, can be set via environment)
        """
        self.api_key = api_key or os.getenv("CIVITAI_API_KEY") or os.getenv("CIVITAI_TOKEN")
        self.base_url = "https://civitai.com/api"
        self.session = requests.Session()
        
        # Set up headers
        self.session.headers.update({
            "User-Agent": "Content-Creator/1.0",
            "Content-Type": "application/json"
        })
        
        # Add authorization if API key is available
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}"
            })
            logger.info("Civitai client initialized with API key")
        else:
            logger.warning("Civitai client initialized without API key - some models may not be accessible")
    
    def search_models(
        self, 
        query: str, 
        model_type: str = None,
        limit: int = 20,
        sort: str = "Highest Rated",
        period: str = "AllTime"
    ) -> List[Dict[str, Any]]:
        """Search for models on Civitai
        
        Args:
            query: Search query
            model_type: Type of model (e.g., "Checkpoint", "LORA", "TextualInversion")
            limit: Maximum number of results
            sort: Sort order ("Most Downloaded", "Highest Rated", "Newest")
            period: Time period ("Day", "Week", "Month", "Year", "AllTime")
            
        Returns:
            List of model information dictionaries
        """
        try:
            params = {
                "query": query,
                "limit": limit,
                "sort": sort,
                "period": period
            }
            
            if model_type:
                params["types"] = model_type
            
            response = self.session.get(f"{self.base_url}/v1/models", params=params)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for item in data.get("items", []):
                model_info = self._parse_model_info(item)
                if model_info:
                    models.append(model_info)
            
            logger.info(f"Found {len(models)} models on Civitai for query: {query}")
            return models
            
        except Exception as e:
            logger.error(f"Error searching Civitai models: {e}")
            return []
    
    def get_model_info(self, model_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model
        
        Args:
            model_id: Civitai model ID
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            response = self.session.get(f"{self.base_url}/v1/models/{model_id}")
            response.raise_for_status()
            
            data = response.json()
            return self._parse_model_info(data)
            
        except Exception as e:
            logger.error(f"Error getting Civitai model info for {model_id}: {e}")
            return None
    
    def get_model_versions(self, model_id: Union[str, int]) -> List[Dict[str, Any]]:
        """Get available versions for a model
        
        Args:
            model_id: Civitai model ID
            
        Returns:
            List of version information
        """
        try:
            response = self.session.get(f"{self.base_url}/v1/models/{model_id}")
            response.raise_for_status()
            
            data = response.json()
            versions = []
            
            for version in data.get("modelVersions", []):
                version_info = {
                    "id": version.get("id"),
                    "name": version.get("name"),
                    "description": version.get("description"),
                    "downloadUrl": version.get("downloadUrl"),
                    "files": version.get("files", []),
                    "images": version.get("images", []),
                    "stats": version.get("stats", {}),
                    "createdAt": version.get("createdAt"),
                    "updatedAt": version.get("updatedAt")
                }
                versions.append(version_info)
            
            return versions
            
        except Exception as e:
            logger.error(f"Error getting model versions for {model_id}: {e}")
            return []
    
    def download_model(
        self,
        model_id: Union[str, int],
        version_id: Optional[Union[str, int]] = None,
        file_type: str = "Model",
        format: str = "SafeTensor",
        size: str = "full",
        fp: str = "fp16",
        output_dir: str = "models",
        progress_callback: Optional[callable] = None
    ) -> Optional[str]:
        """Download a model from Civitai
        
        Args:
            model_id: Civitai model ID
            version_id: Specific version ID (optional, uses latest if not specified)
            file_type: Type of file to download ("Model", "VAE", "Config")
            format: File format ("SafeTensor", "PickleTensor", "Other")
            size: Model size ("full", "pruned")
            fp: Floating point precision ("fp16", "fp32")
            output_dir: Directory to save the model
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to downloaded file or None if failed
        """
        try:
            if progress_callback:
                progress_callback(f"Getting model info for {model_id}...")
            
            # Get model info
            model_info = self.get_model_info(model_id)
            if not model_info:
                logger.error(f"Could not get info for model {model_id}")
                return None
            
            # Get the appropriate version
            versions = model_info.get("modelVersions", [])
            if not versions:
                logger.error(f"No versions found for model {model_id}")
                return None
            
            # Select version
            target_version = None
            if version_id:
                target_version = next((v for v in versions if str(v.get("id")) == str(version_id)), None)
            else:
                target_version = versions[0]  # Use latest version
            
            if not target_version:
                logger.error(f"Version {version_id} not found for model {model_id}")
                return None
            
            # Find the appropriate file
            files = target_version.get("files", [])
            target_file = None
            
            for file in files:
                file_metadata = file.get("metadata", {})
                if (file_metadata.get("type") == file_type and 
                    file_metadata.get("format") == format and
                    file_metadata.get("size") == size and
                    file_metadata.get("fp") == fp):
                    target_file = file
                    break
            
            # If exact match not found, try with more flexible criteria
            if not target_file:
                for file in files:
                    file_metadata = file.get("metadata", {})
                    if file_metadata.get("type") == file_type:
                        target_file = file
                        break
            
            # If still not found, use the first file
            if not target_file and files:
                target_file = files[0]
            
            if not target_file:
                logger.error(f"No suitable file found for model {model_id}")
                return None
            
            # Construct download URL
            download_url = target_file.get("downloadUrl")
            if not download_url:
                # Construct URL manually
                file_id = target_file.get("id")
                download_url = f"{self.base_url}/download/models/{file_id}"
                
                # Add parameters
                params = []
                if file_type != "Model":
                    params.append(f"type={file_type}")
                if format != "SafeTensor":
                    params.append(f"format={format}")
                if size != "full":
                    params.append(f"size={size}")
                if fp != "fp16":
                    params.append(f"fp={fp}")
                
                if params:
                    download_url += "?" + "&".join(params)
            
            # Add API token if available
            if self.api_key:
                separator = "&" if "?" in download_url else "?"
                download_url += f"{separator}token={self.api_key}"
            
            if progress_callback:
                progress_callback(f"Downloading model from Civitai...")
            
            # Download the file
            output_path = self._download_file(
                download_url, 
                model_info.get("name", f"model_{model_id}"), 
                output_dir,
                progress_callback
            )
            
            if output_path:
                logger.info(f"Successfully downloaded Civitai model: {output_path}")
                return output_path
            else:
                logger.error(f"Failed to download model {model_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading Civitai model {model_id}: {e}")
            return None
    
    def _parse_model_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse model information from Civitai API response"""
        try:
            # Get the latest version for additional info
            versions = data.get("modelVersions", [])
            latest_version = versions[0] if versions else {}
            
            # Determine model type for our system
            model_type = data.get("type", "").lower()
            content_type = "image"  # Default
            
            if model_type in ["checkpoint", "lora", "textualinversion", "hypernetwork"]:
                content_type = "image"
            elif model_type in ["poses", "wildcards"]:
                content_type = "other"
            
            # Calculate memory requirement based on model type and files
            memory_requirement = self._estimate_memory_requirement(data)
            
            return {
                "id": data.get("id"),
                "name": data.get("name"),
                "description": data.get("description", ""),
                "type": model_type,
                "content_type": content_type,
                "creator": data.get("creator", {}).get("username", "Unknown"),
                "tags": [tag.get("name") for tag in data.get("tags", [])],
                "stats": data.get("stats", {}),
                "rating": data.get("stats", {}).get("rating", 0),
                "downloads": data.get("stats", {}).get("downloadCount", 0),
                "likes": data.get("stats", {}).get("favoriteCount", 0),
                "comments": data.get("stats", {}).get("commentCount", 0),
                "nsfw": data.get("nsfw", False),
                "allowNoCredit": data.get("allowNoCredit", True),
                "allowCommercialUse": data.get("allowCommercialUse", "None"),
                "allowDerivatives": data.get("allowDerivatives", True),
                "allowDifferentLicense": data.get("allowDifferentLicense", True),
                "modelVersions": versions,
                "latestVersion": latest_version,
                "images": latest_version.get("images", []),
                "memory_requirement": memory_requirement,
                "source": "civitai",
                "url": f"https://civitai.com/models/{data.get('id')}",
                "created_at": data.get("createdAt"),
                "updated_at": data.get("updatedAt")
            }
            
        except Exception as e:
            logger.error(f"Error parsing Civitai model info: {e}")
            return {}
    
    def _estimate_memory_requirement(self, model_data: Dict[str, Any]) -> str:
        """Estimate memory requirement based on model type and size"""
        model_type = model_data.get("type", "").lower()
        
        # Get file sizes from latest version
        versions = model_data.get("modelVersions", [])
        if versions:
            files = versions[0].get("files", [])
            if files:
                # Get largest file size
                max_size = max(file.get("sizeKB", 0) for file in files)
                size_gb = max_size / (1024 * 1024)  # Convert KB to GB
                
                if model_type == "checkpoint":
                    if size_gb > 6:
                        return "12GB"
                    elif size_gb > 3:
                        return "8GB"
                    else:
                        return "6GB"
                elif model_type in ["lora", "textualinversion"]:
                    return "4GB"
                else:
                    return "6GB"
        
        # Default estimates by type
        if model_type == "checkpoint":
            return "8GB"
        elif model_type in ["lora", "textualinversion", "hypernetwork"]:
            return "4GB"
        else:
            return "6GB"
    
    def _download_file(
        self, 
        url: str, 
        model_name: str, 
        output_dir: str,
        progress_callback: Optional[callable] = None
    ) -> Optional[str]:
        """Download a file from URL"""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Start download
            response = self.session.get(url, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            # Get filename from Content-Disposition header or URL
            filename = None
            if "Content-Disposition" in response.headers:
                content_disposition = response.headers["Content-Disposition"]
                if "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip('"')
            
            if not filename:
                # Generate filename
                filename = f"{model_name.replace(' ', '_')}.safetensors"
            
            output_path = Path(output_dir) / filename
            
            # Get file size for progress
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress
            downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded / total_size) * 100
                            progress_callback(f"Downloading: {progress:.1f}%")
            
            if progress_callback:
                progress_callback(f"Download completed: {filename}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading file from {url}: {e}")
            return None
    
    def get_featured_models(self, content_type: str = "image") -> List[Dict[str, Any]]:
        """Get featured/popular models from Civitai
        
        Args:
            content_type: Type of content ("image", "video", etc.)
            
        Returns:
            List of featured model information
        """
        try:
            params = {
                "limit": 20,
                "sort": "Highest Rated",
                "period": "Month"
            }
            
            response = self.session.get(f"{self.base_url}/v1/models", params=params)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for item in data.get("items", []):
                model_info = self._parse_model_info(item)
                if model_info and model_info.get("content_type") == content_type:
                    models.append(model_info)
            
            return models[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Error getting featured Civitai models: {e}")
            return []


# Global Civitai client instance
civitai_client = CivitaiClient() 