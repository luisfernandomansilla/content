"""
Basic Image Generation Example
"""
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from content_creator.image_generator import image_generator


def main():
    """Example of basic image generation"""
    print("🎨 Content Creator - Image Generation Example")
    print("=" * 50)
    
    # Example 1: Basic image generation
    print("\n1. Basic Image Generation:")
    print("-" * 30)
    
    prompt = "A beautiful sunset over a mountain landscape, golden hour lighting, professional photography"
    
    print(f"Prompt: {prompt}")
    print("Generating image...")
    
    def progress_callback(message):
        print(f"  → {message}")
    
    result = image_generator.generate(
        prompt=prompt,
        model_name="FLUX.1-schnell",  # Fast model for demo
        style="Realistic",
        resolution="1024p",
        output_format="PNG",
        quality="Balanced",
        progress_callback=progress_callback
    )
    
    if result:
        print(f"✅ Image generated: {result}")
    else:
        print("❌ Failed to generate image")
    
    # Example 2: Anime style with NSFW model
    print("\n2. Anime Style with Uncensored Model:")
    print("-" * 40)
    
    anime_prompt = "An anime girl with long purple hair, standing in a magical forest, fantasy art style"
    
    print(f"Prompt: {anime_prompt}")
    print("Generating image with uncensored model...")
    
    result2 = image_generator.generate(
        prompt=anime_prompt,
        model_name="Flux-NSFW-uncensored",
        style="Anime",
        resolution="1024p",
        output_format="PNG",
        quality="High",
        negative_prompt="blurry, low quality, distorted",
        guidance_scale=8.0,
        num_inference_steps=32,
        seed=42,
        progress_callback=progress_callback
    )
    
    if result2:
        print(f"✅ Anime image generated: {result2}")
    else:
        print("❌ Failed to generate anime image")
    
    # Example 3: Show available models
    print("\n3. Available Image Models:")
    print("-" * 30)
    
    models = image_generator.get_available_models()
    for name, info in models.items():
        warning = " ⚠️" if info.get('not_for_all_audiences') else ""
        print(f"• {name}{warning}")
        print(f"  Description: {info.get('description', 'No description')}")
        print(f"  Memory: {info.get('memory_requirement', 'Unknown')}")
        print(f"  Max Resolution: {info.get('max_resolution', 'Unknown')}")
        print()
    
    # Example 4: Hardware recommendations
    print("\n4. Hardware Recommendations:")
    print("-" * 30)
    
    recommended = image_generator.get_recommended_models()
    print(f"Recommended models for your hardware: {', '.join(recommended)}")
    
    print("\n🎉 Examples completed!")
    print("\nTo use the GUI interface, run: python main.py")


if __name__ == "__main__":
    main() 