#!/usr/bin/env python3
"""
Test script for Content Creator generation functionality
"""
import os
import sys
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_image_generation():
    """Test image generation functionality"""
    print("\n" + "="*60)
    print("🎨 TESTING IMAGE GENERATION")
    print("="*60)
    
    try:
        from content_creator.image_generator import image_generator
        
        # Test basic image generation
        print("\n1. Testing basic image generation...")
        
        def progress_callback(message):
            print(f"  → {message}")
        
        result = image_generator.generate(
            prompt="A beautiful sunset over mountains",
            model_name="FLUX.1-schnell",
            style="Realistic",
            resolution="1024p",
            output_format="PNG",
            quality="Balanced",
            progress_callback=progress_callback
        )
        
        if result:
            print(f"✅ Image generated successfully: {result}")
        else:
            print("❌ Image generation failed")
        
        # Test available models
        print("\n2. Testing available image models...")
        models = image_generator.get_available_models()
        print(f"Available models: {len(models)}")
        for name, info in list(models.items())[:3]:  # Show first 3
            print(f"  - {name}: {info.get('description', 'No description')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image generation test failed: {e}")
        return False


def test_video_generation():
    """Test video generation functionality"""
    print("\n" + "="*60)
    print("🎬 TESTING VIDEO GENERATION")
    print("="*60)
    
    try:
        from content_creator.generator import generator
        
        # Test basic video generation
        print("\n1. Testing basic video generation...")
        
        def progress_callback(message):
            print(f"  → {message}")
        
        result = generator.generate(
            prompt="A cat walking in a garden",
            model_name="AnimateDiff",
            style="Realistic",
            duration=3,  # Short duration for testing
            resolution="720p",
            fps=24,
            output_format="MP4",
            quality="Balanced",
            progress_callback=progress_callback
        )
        
        if result:
            print(f"✅ Video generated successfully: {result}")
        else:
            print("❌ Video generation failed")
        
        # Test available models
        print("\n2. Testing available video models...")
        models = generator.get_available_models()
        print(f"Available models: {len(models)}")
        for name, info in list(models.items())[:3]:  # Show first 3
            print(f"  - {name}: {info.get('description', 'No description')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Video generation test failed: {e}")
        return False


def test_hardware_detection():
    """Test hardware detection"""
    print("\n" + "="*60)
    print("🖥️  TESTING HARDWARE DETECTION")
    print("="*60)
    
    try:
        from content_creator.utils.hardware import hardware_detector
        
        # Get hardware info
        info = hardware_detector.get_device_info()
        
        print(f"Hardware Type: {info.get('hardware_type', 'Unknown')}")
        print(f"Platform: {info.get('system_info', {}).get('platform', 'Unknown')}")
        print(f"Architecture: {info.get('system_info', {}).get('architecture', 'Unknown')}")
        print(f"GPU Memory: {info.get('gpu_memory_gb', 'Unknown')} GB")
        
        if info.get('hardware_type') == 'apple_silicon':
            print(f"Metal Available: {info.get('metal_available', False)}")
        elif info.get('hardware_type') == 'nvidia_gpu':
            print(f"CUDA Version: {info.get('cuda_version', 'Unknown')}")
            print(f"GPU Name: {info.get('gpu_name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware detection test failed: {e}")
        return False


def test_dependencies():
    """Test required dependencies"""
    print("\n" + "="*60)
    print("📦 TESTING DEPENDENCIES")
    print("="*60)
    
    dependencies = {
        "torch": "PyTorch",
        "PIL": "Pillow",
        "numpy": "NumPy",
        "gradio": "Gradio",
        "huggingface_hub": "Hugging Face Hub"
    }
    
    results = {}
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name} - Available")
            results[module] = True
        except ImportError:
            print(f"❌ {name} - Not available")
            results[module] = False
    
    # Check for optional dependencies
    optional_deps = {
        "diffusers": "Diffusers",
        "transformers": "Transformers", 
        "peft": "PEFT",
        "xformers": "xFormers"
    }
    
    print("\nOptional dependencies:")
    for module, name in optional_deps.items():
        try:
            __import__(module)
            print(f"✅ {name} - Available")
            results[module] = True
        except ImportError:
            print(f"⚠️  {name} - Not available (optional)")
            results[module] = False
    
    return results


def main():
    """Main test function"""
    print("🚀 Content Creator - Generation Test Suite")
    print("=" * 60)
    
    # Test dependencies first
    deps = test_dependencies()
    
    # Test hardware detection
    hardware_ok = test_hardware_detection()
    
    # Test image generation
    image_ok = test_image_generation()
    
    # Test video generation  
    video_ok = test_video_generation()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    print(f"Hardware Detection: {'✅ PASS' if hardware_ok else '❌ FAIL'}")
    print(f"Image Generation: {'✅ PASS' if image_ok else '❌ FAIL'}")
    print(f"Video Generation: {'✅ PASS' if video_ok else '❌ FAIL'}")
    
    # Dependency summary
    print(f"\nCritical Dependencies:")
    critical = ["torch", "PIL", "numpy", "gradio"]
    for dep in critical:
        status = "✅ OK" if deps.get(dep, False) else "❌ MISSING"
        print(f"  {dep}: {status}")
    
    print(f"\nOptional Dependencies:")
    optional = ["diffusers", "transformers", "peft", "xformers"]
    for dep in optional:
        status = "✅ OK" if deps.get(dep, False) else "⚠️  MISSING"
        print(f"  {dep}: {status}")
    
    # Overall status
    overall_ok = hardware_ok and image_ok and video_ok
    print(f"\n🎯 OVERALL STATUS: {'✅ ALL TESTS PASSED' if overall_ok else '❌ SOME TESTS FAILED'}")
    
    if not deps.get("torch", False):
        print("\n⚠️  WARNING: PyTorch not available - generation will use placeholder mode")
    
    if not deps.get("diffusers", False):
        print("⚠️  WARNING: Diffusers not available - real model generation may fail")
    
    print("\n🎉 Test completed!")
    print("To run the full application: python main.py")


if __name__ == "__main__":
    main() 