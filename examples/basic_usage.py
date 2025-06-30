#!/usr/bin/env python3
"""
Basic usage example for Content Creator
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.content_creator import VideoGenerator

def main():
    """Basic example of video generation"""
    
    # Initialize the generator
    generator = VideoGenerator()
    
    # Show hardware info
    print("🖥️ Hardware Information:")
    hardware_info = generator.get_hardware_info()
    print(f"Type: {hardware_info.get('hardware_type')}")
    print(f"GPU Memory: {hardware_info.get('gpu_memory_gb')} GB")
    print()
    
    # List available models
    print("📋 Available Models:")
    models = generator.get_available_models()
    for name, info in list(models.items())[:3]:  # Show first 3
        print(f"- {name}: {info.get('description', 'No description')[:60]}...")
    print()
    
    # Generate a simple video
    print("🎬 Generating video...")
    
    def progress_callback(message):
        print(f"⏳ {message}")
    
    video_path = generator.generate(
        prompt="A beautiful sunset over mountains with clouds moving slowly",
        model_name="AnimateDiff",
        style="Cinematic",
        duration=5,
        resolution="720p",
        quality="Balanced",
        progress_callback=progress_callback
    )
    
    if video_path:
        print(f"✅ Video generated successfully!")
        print(f"📁 Output: {video_path}")
    else:
        print("❌ Failed to generate video")

if __name__ == "__main__":
    main() 