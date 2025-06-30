"""
Main Gradio interface for Content Creator
"""
import os
import logging
import gradio as gr
from typing import List, Optional, Dict, Any, Tuple
import time
from pathlib import Path
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.content_creator import VideoGenerator, ImageGenerator, Config, HardwareDetector
from src.content_creator.models.model_manager import model_manager
from src.content_creator.config import config

# Initialize components
generator = VideoGenerator()
image_generator = ImageGenerator()
hardware_info = generator.get_hardware_info()

# Global variables for UI state
current_model_list = []
search_results = []
generation_status = ""

# State variables for model search
model_details_state = gr.State({})


def get_model_choices():
    """Get list of available models for dropdown"""
    try:
        models = generator.get_available_models()
        choices = []
        for name, info in models.items():
            source = info.get('source', 'default')
            description = info.get('description', '')
            memory = info.get('memory_requirement', 'Unknown')
            choice_text = f"{name} ({source}) - {memory} - {description[:50]}..."
            choices.append((choice_text, name))
        return choices
    except Exception as e:
        logger.error(f"Error getting model choices: {e}")
        return [("AnimateDiff (default)", "AnimateDiff")]


def get_image_model_choices():
    """Get list of available image models for dropdown"""
    try:
        models = image_generator.get_available_models()
        choices = []
        for name, info in models.items():
            description = info.get('description', '')
            memory = info.get('memory_requirement', 'Unknown')
            max_res = info.get('max_resolution', 'Unknown')
            
            # Show warning for NSFW models
            warning = ""
            if info.get('not_for_all_audiences'):
                warning = " ⚠️"
            
            choice_text = f"{name}{warning} - {memory} - {max_res} - {description[:50]}..."
            choices.append((choice_text, name))
        return choices
    except Exception as e:
        logger.error(f"Error getting image model choices: {e}")
        return [("FLUX.1-schnell (default)", "FLUX.1-schnell")]


def get_base_model_choices():
    """Get list of available base models for dropdown (excluding LoRAs)"""
    try:
        models = image_generator.get_available_models()
        base_models = []
        
        for name, info in models.items():
            # Filter out LoRA models from base model selection
            model_type = info.get('type', '').lower()
            is_lora = (
                model_type == 'lora' or 
                'lora' in name.lower() or 
                '(lora)' in name.lower() or
                info.get('requires_base_model', False)
            )
            
            # Only include non-LoRA models as base models
            if not is_lora:
                description = info.get('description', '')
                memory = info.get('memory_requirement', 'Unknown')
                
                # Show warning for NSFW models
                warning = ""
                if info.get('not_for_all_audiences'):
                    warning = " ⚠️"
                
                choice_text = f"{name}{warning} - {memory} - {description[:50]}..."
                base_models.append((choice_text, name))
        
        # Sort alphabetically
        base_models.sort(key=lambda x: x[0])
        
        return base_models
    except Exception as e:
        logger.error(f"Error getting base model choices: {e}")
        return [("FLUX.1-schnell - Default", "FLUX.1-schnell")]


def get_checkpoint_model_choices():
    """Get list of available checkpoint models for dropdown"""
    try:
        models = image_generator.get_available_models()
        checkpoint_models = [("None - No Checkpoint", "none")]  # Add None option
        
        for name, info in models.items():
            # Filter for checkpoint models (LoRAs, checkpoints, etc.)
            model_type = info.get('type', '').lower()
            is_checkpoint = (
                model_type == 'lora' or 
                'lora' in model_type or 
                'lora' in name.lower() or 
                '(lora)' in name.lower() or 
                model_type in ['checkpoint', 'safetensors'] or
                info.get('requires_base_model', False)
            )
            
            if is_checkpoint:
                description = info.get('description', '')
                memory = info.get('memory_requirement', 'Unknown')
                
                # Show warning for NSFW models
                warning = ""
                if info.get('not_for_all_audiences'):
                    warning = " ⚠️"
                
                choice_text = f"{name}{warning} - {memory} - {description[:50]}..."
                checkpoint_models.append((choice_text, name))
        
        # Sort alphabetically (keeping None at the top)
        checkpoint_models[1:] = sorted(checkpoint_models[1:], key=lambda x: x[0])
        
        return checkpoint_models
    except Exception as e:
        logger.error(f"Error getting checkpoint choices: {e}")
        return [("None - No Checkpoint", "none")]


def search_models(query: str, platform: str = "all", limit: int = 20):
    """Search for models across different platforms"""
    try:
        if not query.strip():
            return "Please enter a search query", [], {}
        
        # Determine platforms to search based on filter
        platforms = []
        if platform == "all":
            platforms = ["default", "huggingface", "civitai"]
        elif platform == "huggingface":
            platforms = ["huggingface"]
        elif platform == "civitai":
            platforms = ["civitai"]
        elif platform == "default":
            platforms = ["default"]
        
        results = generator.search_models(query, platforms=platforms, limit=limit)
        
        if not results:
            return f"No models found for query: {query} on {platform}", [], {}
        
        # Format results for HTML display
        formatted_html = f"<h4>Found {len(results)} models</h4>"
        
        # Prepare choices for dropdown and detailed info
        model_choices = []
        model_details = {}
        
        for i, model in enumerate(results[:limit]):
            name = model.get('name', 'Unknown')
            source = model.get('source', 'unknown')
            description = model.get('description', 'No description')
            downloads = model.get('downloads', 0)
            memory = model.get('memory_requirement', 'Unknown')
            model_type = model.get('type', 'unknown')
            model_id = model.get('model_id', model.get('id', name))
            
            # Create unique display name with source
            display_name = f"{name} ({source})"
            model_choices.append(display_name)
            
            # Store detailed info
            model_details[display_name] = {
                'name': name,
                'source': source,
                'description': description,
                'downloads': downloads,
                'memory_requirement': memory,
                'type': model_type,
                'model_id': model_id,
                'full_info': model
            }
            
            # Create HTML card for display
            badges = []
            if source == "civitai":
                if model.get('nsfw'):
                    badges.append("🔞 NSFW")
                if model.get('rating', 0) > 4:
                    badges.append(f"⭐ {model.get('rating', 0):.1f}")
                badges.append(f"❤️ {model.get('likes', 0)}")
            elif source == "huggingface":
                badges.append(f"📥 {downloads:,}")
            
            badges.append(f"💾 {memory}")
            badges.append(f"🏷️ {model_type}")
            
            formatted_html += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px;">
                <h4>{name} <small>({source})</small></h4>
                <p>{" | ".join(badges)}</p>
                <p>{description[:200]}{'...' if len(description) > 200 else ''}</p>
                {f'<p><strong>Creator:</strong> {model.get("creator")}</p>' if source == "civitai" and model.get("creator") else ''}
            </div>
            """
        
        platform_text = "all platforms" if platform == "all" else platform
        return formatted_html, model_choices, model_details
        
    except Exception as e:
        logger.error(f"Error searching models: {e}")
        return f"Error searching models: {e}", [], {}


def download_model(model_name: str, model_type: str, model_details: dict, progress=gr.Progress()):
    """Download a model from Hugging Face or Civitai"""
    try:
        if not model_name:
            return "Please select a model to download", gr.update(), gr.update(), gr.update()
        
        if not model_details:
            return "No model details available. Please search for models first.", gr.update(), gr.update(), gr.update()
        
        # Get model info from details
        model_info = model_details.get(model_name, {})
        if not model_info:
            return f"Model details not found for {model_name}", gr.update(), gr.update(), gr.update()
        
        actual_name = model_info.get('name', model_name)
        source = model_info.get('source', 'unknown')
        model_id = model_info.get('model_id', actual_name)
        
        def progress_callback(msg):
            progress(0.5, desc=msg)
        
        progress(0.1, desc=f"Starting download of {actual_name} from {source}...")
        
        # Determine download parameters based on model type
        download_params = {
            'model_name': actual_name,
            'model_id': model_id,
            'progress_callback': progress_callback
        }
        
        # Add model type hint for organization
        download_params['model_type_hint'] = model_type
        
        result = model_manager.download_model(**download_params)
        
        if result:
            progress(1.0, desc="Download completed!")
            
            # Add to appropriate model list based on type
            success_msg = f"✅ Successfully downloaded {actual_name} from {source}"
            if model_type == "video":
                success_msg += f"\n📹 Model added to video generation options"
                # Refresh video model dropdown
                new_video_choices = get_model_choices()
                return success_msg, gr.update(choices=new_video_choices), gr.update(), gr.update()
            elif model_type == "image":
                success_msg += f"\n🖼️ Model added to image generation options"
                # Refresh image model dropdowns
                new_base_choices = get_base_model_choices()
                new_lora_choices = get_checkpoint_model_choices()
                return success_msg, gr.update(), gr.update(choices=new_base_choices), gr.update(choices=new_lora_choices)
            
            return success_msg, gr.update(), gr.update(), gr.update()
        else:
            return f"❌ Failed to download {actual_name}", gr.update(), gr.update(), gr.update()
            
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return f"❌ Error downloading model: {e}", gr.update(), gr.update(), gr.update()


def refresh_model_dropdowns():
    """Refresh all model dropdowns including base models and LoRAs"""
    try:
        video_choices = get_model_choices()
        base_model_choices = get_base_model_choices()
        lora_choices = get_checkpoint_model_choices()
        
        return (
            gr.update(choices=video_choices),
            gr.update(choices=base_model_choices),
            gr.update(choices=lora_choices),
            "✅ Model lists refreshed successfully! New downloaded models should now be visible."
        )
    except Exception as e:
        logger.error(f"Error refreshing model dropdowns: {e}")
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            f"❌ Error refreshing model lists: {e}"
        )


def generate_video(
    prompt: str,
    model_name: str,
    style: str,
    duration: int,
    resolution: str,
    fps: int,
    output_format: str,
    quality: str,
    reference_images: Optional[List[str]] = None,
    progress=gr.Progress()
):
    """Generate video with progress tracking"""
    try:
        if not prompt.strip():
            return None, "Please enter a prompt"
        
        # Enhanced progress tracking
        current_step = [0]
        total_steps = 4  # Initialize, Download/Check, Process, Generate, Save
        
        def progress_callback(msg):
            if "Downloading" in msg or "searching" in msg:
                current_step[0] = 1
                progress(0.25, desc=msg)
            elif "Processing" in msg:
                current_step[0] = 2
                progress(0.5, desc=msg)
            elif "Generating" in msg:
                current_step[0] = 3
                progress(0.75, desc=msg)
            elif "Saving" in msg:
                current_step[0] = 4
                progress(0.9, desc=msg)
            else:
                # Default progress based on current step
                step_progress = current_step[0] / total_steps
                progress(step_progress, desc=msg)
        
        progress(0.1, desc="Starting video generation...")
        
        # Generate video
        video_path = generator.generate(
            prompt=prompt,
            model_name=model_name,
            style=style,
            duration=duration,
            resolution=resolution,
            fps=fps,
            output_format=output_format,
            quality=quality,
            reference_images=reference_images,
            progress_callback=progress_callback
        )
        
        if video_path:
            progress(1.0, desc="Video generation completed!")
            # Ensure the video path exists and is accessible to Gradio
            if Path(video_path).exists():
                return video_path, f"✅ Video generated successfully: {Path(video_path).name}"
            else:
                return None, f"❌ Generated video file not found: {video_path}"
        else:
            return None, "❌ Failed to generate video"
            
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        return None, f"❌ Error generating video: {e}"


def generate_image(
    prompt: str,
    base_model_name: str,
    checkpoint1_name: str,
    checkpoint2_name: str,
    checkpoint3_name: str,
    style: str,
    resolution: str,
    output_format: str,
    quality: str,
    reference_images: Optional[List[str]] = None,
    negative_prompt: str = "",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 28,
    seed: Optional[int] = None,
    progress=gr.Progress()
):
    """Generate image with progress tracking and separate base/LoRA models"""
    
    try:
        # Debug logging
        logger.info(f"🎯 Generation request received:")
        logger.info(f"   Base model: '{base_model_name}'")
        logger.info(f"   Checkpoint 1: '{checkpoint1_name}'")
        logger.info(f"   Checkpoint 2: '{checkpoint2_name}'")
        logger.info(f"   Checkpoint 3: '{checkpoint3_name}'")
        logger.info(f"   Style: '{style}'")
        logger.info(f"   Prompt: '{prompt[:50]}...'")
        
        # Collect active checkpoints
        active_checkpoints = []
        if checkpoint1_name and checkpoint1_name != "none":
            active_checkpoints.append(checkpoint1_name)
        if checkpoint2_name and checkpoint2_name != "none":
            active_checkpoints.append(checkpoint2_name)
        if checkpoint3_name and checkpoint3_name != "none":
            active_checkpoints.append(checkpoint3_name)
        
        # Determine which model to use
        if active_checkpoints:
            # Use primary checkpoint as main model for now
            # TODO: Implement actual multiple checkpoint loading
            model_name = active_checkpoints[0]
            logger.info(f"✅ Using primary checkpoint: {model_name} with base: {base_model_name}")
            if len(active_checkpoints) > 1:
                logger.info(f"📋 Additional checkpoints: {', '.join(active_checkpoints[1:])}")
        else:
            # Use base model only
            model_name = base_model_name
            logger.info(f"✅ Using base model only: {base_model_name}")
        
        def progress_callback(msg):
            progress(0.5, desc=msg)
        
        progress(0.1, desc="Starting image generation...")
        
        # For now, use the primary checkpoint (first active one)
        primary_checkpoint = active_checkpoints[0] if active_checkpoints else "none"
        
        result = image_generator.generate_with_separate_models(
            prompt=prompt,
            base_model_name=base_model_name,
            lora_model_name=primary_checkpoint,
            style=style,
            resolution=resolution,
            output_format=output_format,
            quality=quality,
            reference_images=reference_images,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            progress_callback=progress_callback
        )
        
        progress(1.0, desc="Image generation completed!")
        
        if result:
            model_display = f"{model_name}"
            if active_checkpoints:
                checkpoint_names = ", ".join(active_checkpoints)
                model_display = f"{checkpoint_names} (Checkpoint{'s' if len(active_checkpoints) > 1 else ''}) + {base_model_name} (base)"
            
            # Ensure the result path exists and is accessible to Gradio
            if Path(result).exists():
                return (
                    result,  # Return the file path directly for Gradio Image component
                    f"✅ Image generated successfully using {model_display}"
                )
            else:
                return (
                    None,
                    f"❌ Generated image file not found: {result}"
                )
        else:
            return None, "❌ Failed to generate image"
            
    except Exception as e:
        logger.error(f"❌ Error generating image: {e}")
        return None, f"❌ Error generating image: {e}"


def get_hardware_info_display():
    """Get formatted hardware information for display"""
    try:
        info = generator.get_hardware_info()
        
        hardware_text = f"""
        **Hardware Information:**
        - **Type:** {info.get('hardware_type', 'Unknown')}
        - **Platform:** {info.get('system_info', {}).get('platform', 'Unknown')}
        - **Architecture:** {info.get('system_info', {}).get('architecture', 'Unknown')}
        - **GPU Memory:** {info.get('gpu_memory_gb', 'Unknown')} GB
        """
        
        if info.get('hardware_type') == 'apple_silicon':
            hardware_text += f"\n- **Metal Available:** {info.get('metal_available', False)}"
        elif info.get('hardware_type') == 'nvidia_gpu':
            hardware_text += f"\n- **CUDA Version:** {info.get('cuda_version', 'Unknown')}"
            hardware_text += f"\n- **GPU Name:** {info.get('gpu_name', 'Unknown')}"
        
        # Add recommended models
        recommended = generator.get_recommended_models()
        if recommended:
            hardware_text += f"\n\n**Recommended Models:** {', '.join(recommended)}"
        
        return hardware_text
        
    except Exception as e:
        logger.error(f"Error getting hardware info: {e}")
        return "Error getting hardware information"


def get_default_base_model():
    """Get the default base model, ensuring it exists in available models"""
    try:
        # First try the configured default
        base_choices = get_base_model_choices()
        
        # Check if the configured default exists in the choices
        for display, value in base_choices:
            if value == config.DEFAULT_IMAGE_MODEL:
                return config.DEFAULT_IMAGE_MODEL
        
        # If configured default not found, return the first non-NSFW model
        for display, value in base_choices:
            if "nsfw" not in value.lower() and "uncensored" not in value.lower():
                return value
        
        # Fallback to first model in list
        if base_choices:
            return base_choices[0][1]
        
        return "FLUX.1-schnell"  # Final fallback
    except Exception as e:
        logger.error(f"Error getting default base model: {e}")
        return "FLUX.1-schnell"


def get_gallery_files():
    """Get all generated images and videos for gallery"""
    try:
        outputs_dir = Path(config.OUTPUT_DIR)
        if not outputs_dir.exists():
            return []
        
        # Get all media files
        media_files = []
        for pattern in ['*.png', '*.jpg', '*.jpeg', '*.mp4', '*.webm', '*.gif']:
            media_files.extend(outputs_dir.glob(pattern))
        
        # Sort by modification time (newest first)
        media_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Return as list of file paths (Gradio can handle Path objects)
        return [str(f) for f in media_files]
    except Exception as e:
        logger.error(f"Error getting gallery files: {e}")
        return []


def get_gallery_info():
    """Get gallery statistics"""
    try:
        files = get_gallery_files()
        total_files = len(files)
        
        # Count by type
        images = len([f for f in files if any(f.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])])
        videos = len([f for f in files if any(f.endswith(ext) for ext in ['.mp4', '.webm', '.gif'])])
        
        # Calculate total size
        total_size = 0
        for file_path in files:
            try:
                total_size += Path(file_path).stat().st_size
            except:
                pass
        
        size_mb = total_size / (1024 * 1024)
        
        return f"📊 **Galería**: {total_files} archivos ({images} imágenes, {videos} videos) - {size_mb:.1f} MB"
    except Exception as e:
        logger.error(f"Error getting gallery info: {e}")
        return "📊 **Galería**: Error obteniendo información"


def delete_file(file_path: str):
    """Delete a file from the gallery"""
    try:
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()
            return f"✅ Archivo eliminado: {Path(file_path).name}", get_gallery_files(), get_gallery_info()
        else:
            return "❌ Archivo no encontrado", get_gallery_files(), get_gallery_info()
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return f"❌ Error eliminando archivo: {e}", get_gallery_files(), get_gallery_info()


def clear_gallery():
    """Clear all files from gallery"""
    try:
        files = get_gallery_files()
        deleted_count = 0
        
        for file_path in files:
            try:
                Path(file_path).unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {e}")
        
        return f"✅ {deleted_count} archivos eliminados de la galería", [], get_gallery_info()
    except Exception as e:
        logger.error(f"Error clearing gallery: {e}")
        return f"❌ Error limpiando galería: {e}", get_gallery_files(), get_gallery_info()


def analyze_prompt_complexity(prompt: str) -> Dict[str, Any]:
    """Analyze prompt complexity and recommend optimal models"""
    word_count = len(prompt.split())
    char_count = len(prompt)
    
    # Rough token estimate (1.3 tokens per word for English)
    estimated_tokens = int(word_count * 1.3)
    
    complexity = {
        'word_count': word_count,
        'char_count': char_count,
        'estimated_tokens': estimated_tokens,
        'is_long': estimated_tokens > 70,
        'is_very_long': estimated_tokens > 100,
        'clip_truncated': estimated_tokens > 77,
        'recommended_models': [],
        'warnings': []
    }
    
    if complexity['clip_truncated']:
        complexity['warnings'].append(
            "⚠️ Este prompt puede ser truncado por modelos basados en CLIP (límite: 77 tokens)"
        )
        complexity['recommended_models'] = [
            "FLUX.1-dev", "FLUX.1-schnell", "Flux-NSFW-uncensored"
        ]
    
    if complexity['is_very_long']:
        complexity['warnings'].append(
            "📝 Prompt muy largo - considera usar modelos FLUX para mejor resultado"
        )
    
    return complexity


def create_gradio_interface():
    """Create the main Gradio interface"""
    
    # Get initial choices
    model_choices = get_model_choices()
    video_style_choices = [(f"{k} - {v['description']}", k) for k, v in config.VIDEO_STYLES.items()]
    image_style_choices = [(f"{k} - {v['description']}", k) for k, v in config.IMAGE_STYLES.items()]
    resolution_choices = [(f"{k} ({v[0]}x{v[1]})", k) for k, v in config.RESOLUTIONS.items()]
    format_choices = [(f"{k} - {v['description']}", k) for k, v in config.OUTPUT_FORMATS.items()]
    quality_choices = [(f"{k} - {v['description']}", k) for k, v in config.QUALITY_PRESETS.items()]
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .model-info {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .hardware-info {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(
        title="Content Creator - AI Video Generation",
        theme=gr.themes.Soft(),
        css=css
    ) as interface:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎬 Content Creator</h1>
            <p>AI-powered video and image generation using vLLM and Gradio</p>
            <p>Create videos and images from text prompts with customizable styles and reference images</p>
        </div>
        """)
        
        with gr.Tabs():
            
            # Main Generation Tab
            with gr.Tab("🎥 Generate Video"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Input Section
                        gr.HTML("<h3>📝 Input Settings</h3>")
                        
                        prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Describe the video you want to generate...",
                            lines=3,
                            value=""
                        )
                        
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=model_choices,
                                label="Model",
                                value=config.DEFAULT_MODEL,
                                interactive=True
                            )
                            
                            style_dropdown = gr.Dropdown(
                                choices=video_style_choices,
                                label="Style",
                                value=config.DEFAULT_STYLE,
                                interactive=True
                            )
                        
                        # Reference Images
                        reference_images = gr.File(
                            label="Reference Images (Optional)",
                            file_count="multiple",
                            file_types=["image"],
                            height=150
                        )
                        
                        # Video Settings
                        gr.HTML("<h3>⚙️ Video Settings</h3>")
                        
                        with gr.Row():
                            duration = gr.Slider(
                                minimum=config.MIN_DURATION,
                                maximum=config.MAX_DURATION,
                                value=config.DEFAULT_DURATION,
                                step=1,
                                label="Duration (seconds)"
                            )
                            
                            fps = gr.Slider(
                                minimum=15,
                                maximum=60,
                                value=config.DEFAULT_FRAME_RATE,
                                step=1,
                                label="FPS"
                            )
                        
                        with gr.Row():
                            resolution = gr.Dropdown(
                                choices=resolution_choices,
                                label="Resolution",
                                value=config.DEFAULT_RESOLUTION,
                                interactive=True
                            )
                            
                            output_format = gr.Dropdown(
                                choices=format_choices,
                                label="Output Format",
                                value=config.DEFAULT_OUTPUT_FORMAT,
                                interactive=True
                            )
                        
                        quality = gr.Dropdown(
                            choices=quality_choices,
                            label="Quality Preset",
                            value=config.DEFAULT_QUALITY,
                            interactive=True
                        )
                        
                        # Generate Button
                        generate_btn = gr.Button(
                            "🚀 Generate Video",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        # Output Section
                        gr.HTML("<h3>📤 Output</h3>")
                        
                        status_text = gr.Textbox(
                            label="Status",
                            value="Ready to generate",
                            interactive=False
                        )
                        
                        output_video = gr.Video(
                            label="Generated Video",
                            height=300
                        )
                        
                        # Hardware Info
                        hardware_display = gr.Markdown(
                            value=get_hardware_info_display(),
                            elem_classes=["hardware-info"]
                        )
            
            # Image Generation Tab
            with gr.Tab("🎨 Generate Image"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Input Section
                        gr.HTML("<h3>📝 Input Settings</h3>")
                        
                        img_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Describe the image you want to generate...",
                            lines=3,
                            value=""
                        )
                        
                        img_negative_prompt = gr.Textbox(
                            label="Negative Prompt (Optional)",
                            placeholder="What you don't want in the image...",
                            lines=2,
                            value=""
                        )
                        
                        # Prompt analysis display
                        prompt_analysis = gr.Markdown(
                            value="💡 **Tip**: Prompts largos funcionan mejor con modelos FLUX",
                            elem_classes=["model-info"],
                            visible=False
                        )
                        
                        with gr.Row():
                            img_base_model_dropdown = gr.Dropdown(
                                choices=get_base_model_choices(),
                                label="Base Model",
                                value=get_default_base_model(),
                                interactive=True,
                                info="Select the main model for generation"
                            )
                            
                            img_checkpoint1_dropdown = gr.Dropdown(
                                choices=get_checkpoint_model_choices(),
                                label="Primary Checkpoint",
                                value="none",
                                interactive=True,
                                info="Optional: Select primary Checkpoint to enhance the base model"
                            )
                        
                        with gr.Row():
                            img_checkpoint2_dropdown = gr.Dropdown(
                                choices=get_checkpoint_model_choices(),
                                label="Secondary Checkpoint",
                                value="none",
                                interactive=True,
                                info="Optional: Select secondary Checkpoint for additional effects"
                            )
                            
                            img_checkpoint3_dropdown = gr.Dropdown(
                                choices=get_checkpoint_model_choices(),
                                label="Tertiary Checkpoint",
                                value="none",
                                interactive=True,
                                info="Optional: Select tertiary Checkpoint for fine-tuning"
                            )
                        
                        with gr.Row():
                            img_style_dropdown = gr.Dropdown(
                                choices=image_style_choices,
                                label="Style",
                                value=config.DEFAULT_STYLE,
                                interactive=True
                            )
                        
                        # Reference Images
                        img_reference_images = gr.File(
                            label="Reference Images (Optional)",
                            file_count="multiple",
                            file_types=["image"],
                            height=150
                        )
                        
                        # Image Settings
                        gr.HTML("<h3>⚙️ Image Settings</h3>")
                        
                        with gr.Row():
                            img_resolution = gr.Dropdown(
                                choices=resolution_choices,
                                label="Resolution",
                                value=config.DEFAULT_RESOLUTION,
                                interactive=True
                            )
                            
                            img_output_format = gr.Dropdown(
                                choices=[
                                    ("PNG - High quality with transparency", "PNG"),
                                    ("JPEG - Standard image format", "JPEG"),
                                    ("WebP - Modern efficient format", "WEBP"),
                                    ("TIFF - Professional format", "TIFF")
                                ],
                                label="Output Format",
                                value="PNG",
                                interactive=True
                            )
                        
                        with gr.Row():
                            img_quality = gr.Dropdown(
                                choices=quality_choices,
                                label="Quality Preset",
                                value=config.DEFAULT_QUALITY,
                                interactive=True
                            )
                            
                            img_seed = gr.Number(
                                label="Seed (Optional)",
                                value=None,
                                precision=0,
                                minimum=0,
                                maximum=2147483647
                            )
                        
                        # Advanced Settings
                        with gr.Accordion("🔧 Advanced Settings", open=False):
                            with gr.Row():
                                img_guidance_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=20.0,
                                    value=7.5,
                                    step=0.5,
                                    label="Guidance Scale"
                                )
                                
                                img_inference_steps = gr.Slider(
                                    minimum=10,
                                    maximum=100,
                                    value=28,
                                    step=1,
                                    label="Inference Steps"
                                )
                        
                        # Generate Button
                        img_generate_btn = gr.Button(
                            "🎨 Generate Image",
                            variant="primary",
                            size="lg"
                        )
                        
                        # NSFW Warning
                        gr.HTML("""
                        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107;">
                            <strong>⚠️ Content Warning:</strong>
                            <p>Some models may generate NSFW content. Models marked with ⚠️ are uncensored models. Use responsibly and in accordance with your local laws and community guidelines.</p>
                        </div>
                        """)
                        
                        # CLIP Limitation Info
                        gr.HTML("""
                        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #28a745;">
                            <strong>🔤 Prompts Largos - ✅ COMPEL DISPONIBLE:</strong>
                            <p><strong>CLIP (77 tokens límite):</strong> Stable Diffusion, SDXL, etc.</p>
                            <p><strong>T5 (sin límite práctico):</strong> FLUX.1-dev, FLUX.1-schnell</p>
                            <p><strong>✨ Compel activo:</strong> Convierte prompts largos en embeddings avanzados para modelos CLIP</p>
                            <p>🎯 <strong>Ahora puedes usar prompts largos con cualquier modelo</strong> - Compel manejará automáticamente la limitación de CLIP</p>
                        </div>
                        """)
                    
                    with gr.Column(scale=1):
                        # Output Section
                        gr.HTML("<h3>📤 Output</h3>")
                        
                        img_status_text = gr.Textbox(
                            label="Status",
                            value="Ready to generate",
                            interactive=False
                        )
                        
                        img_output = gr.Image(
                            label="Generated Image",
                            height=400,
                            show_download_button=True
                        )
                        
                        # Model Info
                        img_model_info = gr.Markdown(
                            value="Select a model to see details",
                            elem_classes=["model-info"]
                        )
            
            # Model Browser Tab
            with gr.Tab("🔍 Browse Models"):
                gr.HTML("<h3>Search and Download Models from Hugging Face and Civitai</h3>")
                
                with gr.Row():
                    search_query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter keywords to search for models...",
                        scale=2
                    )
                    
                    platform_filter = gr.Dropdown(
                        choices=[
                            ("All Platforms", "all"),
                            ("Hugging Face", "huggingface"),
                            ("Civitai", "civitai"),
                            ("Default Models", "default")
                        ],
                        label="Platform",
                        value="all",
                        scale=1
                    )
                    
                    search_btn = gr.Button("🔍 Search", scale=1)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        search_results_display = gr.HTML(
                            value="<p>Enter a search query to find models</p>"
                        )
                    
                    with gr.Column(scale=1):
                        # Model selection section
                        gr.HTML("<h4>📦 Model Selection</h4>")
                        
                        available_models = gr.Dropdown(
                            label="Available Models",
                            choices=[],
                            value=None,
                            interactive=True,
                            info="Select a model from search results"
                        )
                        
                        model_type = gr.Dropdown(
                            label="Model Type",
                            choices=[
                                ("🎥 Video Generation", "video"),
                                ("🖼️ Image Generation", "image")
                            ],
                            value="video",
                            interactive=True,
                            info="Specify how you want to use this model"
                        )
                        
                        # Model details display
                        model_details_display = gr.Markdown(
                            value="Select a model to see details",
                            elem_classes=["model-info"]
                        )
                        
                        download_btn = gr.Button(
                            "📥 Download Model",
                            variant="primary",
                            size="lg"
                        )
                        
                        download_status = gr.Textbox(
                            label="Download Status",
                            value="No downloads in progress",
                            interactive=False
                        )
                        
                        # Add refresh button for model lists
                        gr.HTML("<h4>🔄 Model List Management</h4>")
                        refresh_models_btn = gr.Button(
                            "🔄 Refresh Model Lists",
                            variant="secondary",
                            size="sm"
                        )
                        refresh_status = gr.Textbox(
                            label="Refresh Status",
                            value="Click to refresh if new models don't appear",
                            interactive=False,
                            lines=2
                        )
                
                # Featured Models Section
                gr.HTML("<h4>⭐ Featured Models</h4>")
                
                featured_models = gr.HTML(
                    value="""
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 8px;">
                            <h4>AnimateDiff</h4>
                            <p>Animation-focused video generation with motion control</p>
                            <small>Memory: 8GB | Default Model</small>
                        </div>
                        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 8px;">
                            <h4>Stable Video Diffusion</h4>
                            <p>High-quality video generation from images</p>
                            <small>Memory: 12GB | Hugging Face</small>
                        </div>
                        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 8px;">
                            <h4>Text2Video-Zero</h4>
                            <p>Text-to-video generation model</p>
                            <small>Memory: 6GB | Hugging Face</small>
                        </div>
                    </div>
                    """
                )
            
            # Settings Tab
            with gr.Tab("⚙️ Settings"):
                gr.HTML("<h3>Configuration</h3>")
                
                with gr.Row():
                    with gr.Column():
                        # API Keys Section
                        gr.HTML("<h4>🔑 API Keys</h4>")
                        
                        hf_token = gr.Textbox(
                            label="Hugging Face Token",
                            placeholder="Enter your HF token for private models and faster downloads",
                            type="password",
                            value=os.getenv("HUGGINGFACE_API_KEY", "")
                        )
                        
                        civitai_token = gr.Textbox(
                            label="Civitai API Key",
                            placeholder="Enter your Civitai API key for downloading models",
                            type="password",
                            value=os.getenv("CIVITAI_API_KEY", "")
                        )
                        
                        with gr.Row():
                            save_hf_token_btn = gr.Button("💾 Save HF Token")
                            save_civitai_token_btn = gr.Button("💾 Save Civitai Key")
                        
                        # Environment Variables Info
                        gr.HTML("""
                        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        <h5>Environment Variables</h5>
                        <p>You can also set these environment variables:</p>
                        <ul>
                            <li><code>HUGGINGFACE_API_KEY</code> or <code>HF_TOKEN</code> - Your Hugging Face API key</li>
                            <li><code>CIVITAI_API_KEY</code> or <code>CIVITAI_TOKEN</code> - Your Civitai API key</li>
                        </ul>
                        <p><strong>How to get API keys:</strong></p>
                        <ul>
                            <li>🤗 <a href="https://huggingface.co/settings/tokens" target="_blank">Hugging Face Tokens</a></li>
                            <li>🎨 <a href="https://civitai.com/user/account" target="_blank">Civitai API Settings</a></li>
                        </ul>
                        </div>
                        """)
                    
                    with gr.Column():
                        # Cache Management
                        gr.HTML("<h4>🗂️ Cache Management</h4>")
                        
                        cache_info = gr.Textbox(
                            label="Cache Information",
                            value="Loading cache info...",
                            interactive=False,
                            lines=3
                        )
                        
                        with gr.Row():
                            refresh_cache_btn = gr.Button("🔄 Refresh Cache Info")
                            clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
                        
                        cache_status = gr.Textbox(
                            label="Cache Status",
                            value="Ready",
                            interactive=False
                        )
            
            # Gallery Tab
            with gr.Tab("🖼️ Galería"):
                gr.HTML("<h3>📁 Archivos Generados</h3>")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Gallery display
                        gallery = gr.Gallery(
                            label="Imágenes y Videos Generados",
                            value=get_gallery_files(),
                            columns=3,
                            rows=3,
                            height="600px",
                            object_fit="contain",
                            show_download_button=True,
                            show_share_button=False,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        # Gallery controls
                        gr.HTML("<h4>🎛️ Controles</h4>")
                        
                        gallery_info = gr.Markdown(
                            value=get_gallery_info(),
                            elem_classes=["model-info"]
                        )
                        
                        refresh_gallery_btn = gr.Button(
                            "🔄 Refrescar Galería",
                            variant="secondary",
                            size="sm"
                        )
                        
                        clear_gallery_btn = gr.Button(
                            "🗑️ Limpiar Galería",
                            variant="secondary",
                            size="sm"
                        )
                        
                        gallery_status = gr.Textbox(
                            label="Estado",
                            value="Galería lista",
                            interactive=False,
                            lines=2
                        )
                        
                        # File selection for deletion
                        gr.HTML("<h4>🗂️ Gestión de Archivos</h4>")
                        
                        selected_file = gr.Textbox(
                            label="Archivo Seleccionado",
                            placeholder="Haz click en una imagen para seleccionarla",
                            interactive=True
                        )
                        
                        delete_selected_btn = gr.Button(
                            "🗑️ Eliminar Seleccionado",
                            variant="secondary",
                            size="sm"
                        )
                        
                        # Tips
                        gr.HTML("""
                        <div style="background-color: #e7f3ff; padding: 10px; border-radius: 5px; margin: 10px 0;">
                            <h5>💡 Consejos:</h5>
                            <ul>
                                <li>Haz click en cualquier imagen para descargarla</li>
                                <li>Los archivos más recientes aparecen primero</li>
                                <li>Puedes eliminar archivos individuales o toda la galería</li>
                                <li>La galería se actualiza automáticamente al generar nuevos archivos</li>
                            </ul>
                        </div>
                        """)
        
        # State variables for model search
        model_details_state = gr.State({})
        
        # Event Handlers
        
        # Main generation
        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt, model_dropdown, style_dropdown, duration,
                resolution, fps, output_format, quality, reference_images
            ],
            outputs=[output_video, status_text]
        )
        
        # Image generation
        img_generate_btn.click(
            fn=generate_image,
            inputs=[
                img_prompt, img_base_model_dropdown, img_checkpoint1_dropdown,
                img_checkpoint2_dropdown, img_checkpoint3_dropdown,
                img_style_dropdown, img_resolution, img_output_format, img_quality,
                img_reference_images, img_negative_prompt,
                img_guidance_scale, img_inference_steps, img_seed
            ],
            outputs=[img_output, img_status_text]
        )
        
        # Image model info update
        def update_image_model_info(base_model_name, checkpoint1_name, checkpoint2_name, checkpoint3_name):
            """Update image model information display based on selected base and checkpoint models"""
            try:
                info_parts = []
                
                # Base model info
                if base_model_name:
                    base_models = image_generator.get_available_models()
                    base_info = base_models.get(base_model_name, {})
                    
                    info_parts.append(f"**Base Model:** {base_model_name}")
                    if base_info.get('description'):
                        info_parts.append(f"**Description:** {base_info['description']}")
                    if base_info.get('memory_requirement'):
                        info_parts.append(f"**Memory:** {base_info['memory_requirement']}")
                    if base_info.get('max_resolution'):
                        info_parts.append(f"**Max Resolution:** {base_info['max_resolution']}")
                
                # Checkpoint model info
                checkpoint_models = image_generator.get_available_models()
                active_checkpoints = []
                
                for i, checkpoint_name in enumerate([checkpoint1_name, checkpoint2_name, checkpoint3_name], 1):
                    if checkpoint_name and checkpoint_name != "none":
                        active_checkpoints.append(checkpoint_name)
                        checkpoint_info = checkpoint_models.get(checkpoint_name, {})
                        
                        info_parts.append(f"\n**Checkpoint {i}:** {checkpoint_name}")
                        if checkpoint_info.get('description'):
                            info_parts.append(f"**Description:** {checkpoint_info['description']}")
                        if checkpoint_info.get('not_for_all_audiences'):
                            info_parts.append("⚠️ **Content Warning:** This is an uncensored model")
                
                if not active_checkpoints:
                    info_parts.append(f"\n**Checkpoints:** None - Using base model only")
                
                if not info_parts:
                    return "Select models to see details"
                
                return "\n".join(info_parts)
                
            except Exception as e:
                logger.error(f"Error updating model info: {e}")
                return f"Error loading model information: {e}"
        
        # Update model info when any dropdown changes
        for dropdown in [img_base_model_dropdown, img_checkpoint1_dropdown, img_checkpoint2_dropdown, img_checkpoint3_dropdown]:
            dropdown.change(
                fn=update_image_model_info,
                inputs=[img_base_model_dropdown, img_checkpoint1_dropdown, img_checkpoint2_dropdown, img_checkpoint3_dropdown],
                outputs=[img_model_info]
            )
        
        # Model search
        def search_and_update_dropdown(query, platform, limit=20):
            """Search models and return proper dropdown updates"""
            try:
                html_results, model_choices, model_details = search_models(query, platform, limit)
                
                # Update dropdown with new choices and clear selection
                if model_choices:
                    return (
                        html_results,  # HTML display
                        gr.update(choices=model_choices, value=None),  # Updated dropdown
                        model_details  # Model details state
                    )
                else:
                    return (
                        html_results,
                        gr.update(choices=[], value=None),
                        {}
                    )
            except Exception as e:
                error_msg = f"Search error: {e}"
                return (
                    f"<p style='color: red;'>{error_msg}</p>",
                    gr.update(choices=[], value=None),
                    {}
                )
        
        search_btn.click(
            fn=search_and_update_dropdown,
            inputs=[search_query, platform_filter],
            outputs=[search_results_display, available_models, model_details_state]
        )
        
        # Model selection change handler
        def update_model_details(selected_model, model_details):
            if selected_model and model_details and selected_model in model_details:
                info = model_details[selected_model]
                
                details_text = f"""
                **Model:** {info.get('name', 'Unknown')}
                
                **Source:** {info.get('source', 'Unknown')}
                
                **Description:** {info.get('description', 'No description')}
                
                **Memory Requirement:** {info.get('memory_requirement', 'Unknown')}
                
                **Type:** {info.get('type', 'Unknown')}
                
                **Downloads:** {info.get('downloads', 'N/A'):,} (if available)
                """
                
                if info.get('source') == 'civitai':
                    if info.get('full_info', {}).get('creator'):
                        details_text += f"\n**Creator:** {info['full_info']['creator']}"
                    if info.get('full_info', {}).get('nsfw'):
                        details_text += "\n\n⚠️ **Content Warning:** This model may generate NSFW content."
                
                return details_text
            else:
                return "Select a model to see details"
        
        available_models.change(
            fn=update_model_details,
            inputs=[available_models, model_details_state],
            outputs=[model_details_display]
        )
        
        # Model download
        download_btn.click(
            fn=download_model,
            inputs=[available_models, model_type, model_details_state],
            outputs=[download_status, model_dropdown, img_base_model_dropdown, img_checkpoint1_dropdown]
        )
        
        # Model dropdown refresh
        refresh_models_btn.click(
            fn=refresh_model_dropdowns,
            outputs=[model_dropdown, img_base_model_dropdown, img_checkpoint1_dropdown, refresh_status]
        )
        
        # Settings handlers
        def save_hf_token(token):
            if token:
                os.environ["HUGGINGFACE_API_KEY"] = token
                # Update the client with new token
                try:
                    from src.content_creator.models.huggingface_client import hf_client
                    hf_client.api_key = token
                    hf_client._setup_client()
                except:
                    pass
                return "Hugging Face token saved successfully"
            return "No token provided"
        
        def save_civitai_token(token):
            if token:
                os.environ["CIVITAI_API_KEY"] = token
                # Update the client with new token
                try:
                    from src.content_creator.models.civitai_client import civitai_client
                    civitai_client.api_key = token
                    civitai_client.session.headers.update({
                        "Authorization": f"Bearer {token}"
                    })
                except:
                    pass
                return "Civitai API key saved successfully"
            return "No token provided"
        
        save_hf_token_btn.click(
            fn=save_hf_token,
            inputs=[hf_token],
            outputs=[cache_status]
        )
        
        save_civitai_token_btn.click(
            fn=save_civitai_token,
            inputs=[civitai_token],
            outputs=[cache_status]
        )
        
        def get_cache_info():
            try:
                # Get model cache info
                model_info = model_manager.get_cache_info()
                
                # Get pipeline cache info
                image_cache_info = image_generator.get_cache_info()
                video_cache_info = generator.get_video_cache_info()
                
                return f"""📋 Cache Information:

🗄️ Model Downloads:
  • Downloaded models: {model_info['downloaded_models']}
  • Cache file exists: {model_info['cache_file_exists']}
  • Cache file size: {model_info['cache_file_size']} bytes

🖼️ Image Pipelines (Memory):
  • Cached models: {image_cache_info['cached_models']}
  • Current model: {image_cache_info['current_model'] or 'None'}
  • Memory status: {image_cache_info['memory_usage']}

🎬 Video Pipelines (Memory):
  • Cached models: {video_cache_info['cached_models']}
  • Current model: {video_cache_info['current_model'] or 'None'}
  • Memory status: {video_cache_info['memory_usage']}

💡 Pipeline caching speeds up generation by avoiding model reloading"""
            except Exception as e:
                return f"Error getting cache info: {e}"
        
        refresh_cache_btn.click(
            fn=get_cache_info,
            outputs=[cache_info]
        )
        
        def clear_cache():
            try:
                # Clear all types of cache
                model_manager.clear_cache()
                image_generator.clear_pipeline_cache()
                generator.clear_video_pipeline_cache()
                
                return "✅ All caches cleared successfully:\n• Model downloads cache\n• Image pipeline cache\n• Video pipeline cache"
            except Exception as e:
                return f"Error clearing cache: {e}"
        
        clear_cache_btn.click(
            fn=clear_cache,
            outputs=[cache_status]
        )
        
        # Gallery event handlers
        refresh_gallery_btn.click(
            fn=lambda: (get_gallery_files(), get_gallery_info()),
            outputs=[gallery, gallery_info]
        )
        
        clear_gallery_btn.click(
            fn=clear_gallery,
            outputs=[gallery_status, gallery, gallery_info]
        )
        
        delete_selected_btn.click(
            fn=delete_file,
            inputs=[selected_file],
            outputs=[gallery_status, gallery, gallery_info]
        )
        
        # Auto-refresh gallery when new images/videos are generated  
        def auto_refresh_gallery():
            return get_gallery_files(), get_gallery_info()
            
        # Add separate events for gallery refresh after generation
        img_status_text.change(
            fn=auto_refresh_gallery,
            outputs=[gallery, gallery_info]
        )
        
        status_text.change(
            fn=auto_refresh_gallery,
            outputs=[gallery, gallery_info]
        )
        
        # Prompt analysis for long prompt detection
        def update_prompt_analysis(prompt):
            if not prompt or len(prompt.strip()) < 10:
                return gr.update(visible=False)
            
            analysis = analyze_prompt_complexity(prompt)
            
            if analysis['clip_truncated'] or analysis['is_very_long']:
                warnings_text = "\n".join(analysis['warnings'])
                
                if analysis['recommended_models']:
                    recommendations = ", ".join(analysis['recommended_models'])
                    warnings_text += f"\n\n🎯 **Modelos recomendados**: {recommendations}"
                
                warnings_text += f"\n\n📊 **Estadísticas del prompt**:"
                warnings_text += f"\n- Palabras: {analysis['word_count']}"
                warnings_text += f"\n- Tokens estimados: {analysis['estimated_tokens']}"
                warnings_text += f"\n- Límite CLIP: 77 tokens"
                
                return gr.update(value=warnings_text, visible=True)
            else:
                return gr.update(visible=False)
        
        # Connect prompt analysis to input changes
        img_prompt.change(
            fn=update_prompt_analysis,
            inputs=[img_prompt],
            outputs=[prompt_analysis]
        )
        
        # Load initial cache info
        interface.load(
            fn=get_cache_info,
            outputs=[cache_info]
        )
    
    return interface


if __name__ == "__main__":
    # Log configuration info
    config_summary = config.get_config_summary()
    logger.info(f"Starting Content Creator in {config_summary['environment']} mode")
    logger.info(f"Server will run on {config_summary['server']['host']}:{config_summary['server']['port']}")
    logger.info(f"Debug mode: {config_summary['debug']}")
    logger.info(f"Tokens configured - HF: {config_summary['api_tokens']['hf_token_set']}, Civitai: {config_summary['api_tokens']['civitai_token_set']}")
    
    # Create and launch the interface
    interface = create_gradio_interface()
    
    # Launch settings
    interface.launch(
        server_name=config.GRADIO_HOST,
        server_port=config.GRADIO_PORT,
        share=config.GRADIO_SHARE,
        show_error=True
    ) 