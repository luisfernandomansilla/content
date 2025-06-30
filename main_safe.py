#!/usr/bin/env python3
"""
Content Creator - Simplified Launch for Gradio 5.x compatibility
This version avoids complex component configurations that cause JSON schema issues
"""

import gradio as gr
import os
import logging
from typing import List, Optional, Dict, Any

# Core imports
from src.content_creator.config import config
from src.content_creator.generator import generator
from src.content_creator.image_generator import image_generator
from src.content_creator.models.model_manager import model_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple model choices functions
def get_simple_model_choices():
    """Get simplified model choices that work with Gradio 5.x"""
    try:
        models = model_manager.get_available_models()
        return [name for name, info in models.items() if info.get('available', True)]
    except:
        return ["FLUX.1-schnell", "FLUX.1-dev", "stable-diffusion-v1-5"]

def generate_image_simple(
    prompt: str,
    model_name: str,
    style: str = "None",
    resolution: str = "1024x1024",
    progress=gr.Progress()
):
    """Simplified image generation function"""
    try:
        if not prompt or not prompt.strip():
            return None, "❌ Please enter a prompt"
        
        def progress_callback(msg):
            progress(0.5, desc=msg)
        
        logger.info(f"🎨 Generating image with prompt: {prompt[:100]}...")
        
        # Generate using the separate models system
        result = image_generator.generate_with_separate_models(
            prompt=prompt,
            base_model_name=model_name,
            lora_model_name="none",
            style=style,
            resolution=resolution,
            progress_callback=progress_callback
        )
        
        if result:
            return result, "✅ Image generated successfully!"
        else:
            return None, "❌ Failed to generate image"
            
    except Exception as e:
        logger.error(f"Error in simple generation: {e}")
        return None, f"❌ Error: {e}"

def create_simple_interface():
    """Create a simplified Gradio interface"""
    
    with gr.Blocks(
        title="Content Creator - AI Image Generation",
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .generate-btn { background: linear-gradient(45deg, #4CAF50, #45a049) !important; }
        """
    ) as interface:
        
        gr.HTML("""
        <div class="main-container">
            <h1>🎨 Content Creator - AI Image Generation</h1>
            <p>Generate high-quality images using advanced AI models with Compel support for long prompts</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<h3>📝 Input Settings</h3>")
                
                prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=4,
                    max_lines=8
                )
                
                with gr.Row():
                    model_choice = gr.Dropdown(
                        choices=get_simple_model_choices(),
                        label="Model",
                        value="FLUX.1-schnell",
                        interactive=True
                    )
                    
                    style_choice = gr.Dropdown(
                        choices=["None", "Anime", "Realistic", "Artistic", "Hentai"],
                        label="Style",
                        value="None",
                        interactive=True
                    )
                
                resolution_choice = gr.Dropdown(
                    choices=["512x512", "768x768", "1024x1024", "1024x768", "768x1024"],
                    label="Resolution",
                    value="1024x1024",
                    interactive=True
                )
                
                generate_btn = gr.Button(
                    "🎨 Generate Image",
                    variant="primary",
                    elem_classes=["generate-btn"]
                )
                
                # Info about Compel
                gr.HTML("""
                <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #28a745;">
                    <strong>✅ COMPEL DISPONIBLE:</strong>
                    <p>🎯 Los prompts largos se procesan automáticamente con embeddings avanzados</p>
                    <p>🔤 Sin limitación de 77 tokens - usa prompts tan largos como quieras</p>
                </div>
                """)
            
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 Output</h3>")
                
                status = gr.Textbox(
                    label="Status",
                    value="Ready to generate",
                    interactive=False
                )
                
                output_image = gr.Image(
                    label="Generated Image",
                    show_download_button=True,
                    height=400
                )
        
        # Event handlers
        generate_btn.click(
            fn=generate_image_simple,
            inputs=[prompt, model_choice, style_choice, resolution_choice],
            outputs=[output_image, status],
            show_progress=True
        )
    
    return interface

if __name__ == "__main__":
    logger.info("🚀 Starting Content Creator (Safe Mode)")
    logger.info("✅ Compel support enabled for long prompts")
    
    # Create simplified interface
    interface = create_simple_interface()
    
    # Launch with safe settings
    try:
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            enable_queue=True
        )
    except ValueError as e:
        if "shareable link must be created" in str(e):
            logger.warning("⚠️ Creating shareable link...")
            interface.launch(
                share=True,
                debug=False,
                show_error=True
            )
        else:
            raise e
    except Exception as e:
        logger.error(f"Launch failed: {e}")
        # Final fallback
        interface.launch(share=True) 