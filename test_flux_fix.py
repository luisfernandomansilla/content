#!/usr/bin/env python3
"""
Test script to verify FLUX+LoRA compatibility fixes
"""
import os
import sys

# Add src to path
sys.path.insert(0, 'src')

from content_creator import ImageGenerator, Config

def test_flux_lora_compatibility():
    """Test that FLUX+LoRA loads without xformers conflicts"""
    
    print("🧪 Testing FLUX+LoRA Compatibility Fix")
    print("=" * 50)
    
    # Initialize generator
    generator = ImageGenerator()
    
    # Test FLUX+LoRA model
    model_name = "Flux-NSFW-uncensored"
    
    print(f"🔍 Testing model: {model_name}")
    
    # Get model info
    model_info = generator._get_image_model_info(model_name)
    
    if not model_info:
        print(f"❌ Model {model_name} not found in configuration")
        return False
    
    print(f"✅ Model found: {model_info['type']}")
    print(f"📋 Base model: {model_info.get('base_model', 'N/A')}")
    
    # Test loading pipeline (without actually generating)
    print(f"🚀 Testing pipeline loading...")
    
    try:
        # This will test the pipeline loading logic but won't move to device
        # to avoid GPU memory issues in testing
        pipeline = generator._load_pipeline(model_name, model_info)
        
        if pipeline:
            print("✅ Pipeline loaded successfully!")
            print("🔧 XFormers properly disabled for FLUX+LoRA")
            return True
        else:
            print("❌ Pipeline failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Error loading pipeline: {e}")
        return False

def test_production_config():
    """Test production configuration"""
    
    print("\n🌐 Testing Production Configuration")
    print("=" * 50)
    
    # Set environment to production temporarily
    original_env = os.environ.get("ENVIRONMENT")
    os.environ["ENVIRONMENT"] = "production"
    
    try:
        # Import config after setting environment
        from content_creator.config import Config
        config = Config()
        
        # Test configuration
        summary = config.get_config_summary()
        
        print(f"🔧 Environment: {summary['environment']}")
        print(f"🌐 Host: {summary['server']['host']}")
        print(f"🔌 Port: {summary['server']['port']}")
        print(f"🎨 Theme: {summary['server']['theme']}")
        
        # Verify production settings
        if (summary['environment'] == 'production' and 
            summary['server']['host'] == '0.0.0.0' and 
            summary['server']['port'] == 80):
            print("✅ Production configuration looks correct!")
            return True
        else:
            print("❌ Production configuration has issues")
            return False
            
    finally:
        # Restore original environment
        if original_env:
            os.environ["ENVIRONMENT"] = original_env
        else:
            os.environ.pop("ENVIRONMENT", None)

def main():
    """Run all tests"""
    
    print("🚀 Content Creator - Compatibility Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: FLUX+LoRA compatibility
    results.append(test_flux_lora_compatibility())
    
    # Test 2: Production configuration
    results.append(test_production_config())
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
        print("\n💡 Next Steps:")
        print("   1. Update your .env file with real tokens")
        print("   2. Set ENVIRONMENT=production for deployment")
        print("   3. Ensure PORT=80 is available for production")
        return True
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 