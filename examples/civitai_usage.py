"""
Example: Using Civitai models with Content Creator

This example demonstrates how to:
1. Search for models on Civitai
2. Download models from Civitai
3. Use them for generation

Make sure to set your Civitai API key:
export CIVITAI_API_KEY="your_api_key_here"
"""

import os
import sys
sys.path.append('.')

from src.content_creator import generator, image_generator
from src.content_creator.models.civitai_client import civitai_client


def main():
    # Set up Civitai API key if not already set
    if not os.getenv("CIVITAI_API_KEY"):
        print("⚠️  Warning: CIVITAI_API_KEY not set. Some models may require authentication.")
        print("Get your API key from: https://civitai.com/user/account")
        print()
    
    print("🎨 Content Creator - Civitai Integration Example")
    print("=" * 50)
    
    # Example 1: Search for models on Civitai
    print("\n1. 🔍 Searching for anime models on Civitai...")
    print("Note: This requires an internet connection and Civitai API access")
    try:
        # Demo search - may not return results without API key
        anime_models = civitai_client.search_models(
            query="anime",
            model_type="Checkpoint",
            limit=5
        )
        
        if anime_models:
            print(f"Found {len(anime_models)} anime models:")
            for i, model in enumerate(anime_models, 1):
                print(f"  {i}. {model.get('name', 'Unknown')}")
                print(f"     👤 By: {model.get('creator', 'Unknown')}")
                print(f"     📥 Downloads: {model.get('downloads', 0):,}")
                print(f"     ⭐ Rating: {model.get('rating', 0):.1f}")
                if model.get('nsfw'):
                    print(f"     🔞 NSFW Content")
                print()
        else:
            print("No models found (may require API key or internet connection)")
    except Exception as e:
        print(f"Search unavailable: {e}")
    
    # Example 2: Search using the integrated search (all platforms)
    print("\n2. 🌐 Searching across all platforms...")
    try:
        all_results = generator.search_models(
            query="realistic portrait",
            platforms=["huggingface", "civitai"],
            limit=10
        )
        
        print(f"Found {len(all_results)} models across platforms:")
        for model in all_results:
            source = model.get('source', 'unknown')
            name = model.get('name', 'Unknown')
            print(f"  📦 {name} (from {source})")
            
            if source == "civitai":
                print(f"    🆔 Civitai ID: {model.get('civitai_id')}")
                if model.get('nsfw'):
                    print(f"    🔞 NSFW")
    
    except Exception as e:
        print(f"Error in integrated search: {e}")
    
    # Example 3: Get model information
    print("\n3. 📋 Getting detailed model information...")
    print("Demo: How to get model information from Civitai")
    print("Example model IDs:")
    print("  • 25694 - Popular anime checkpoint")
    print("  • 4384 - DreamShaper")
    print("  • 6424 - Anything V3")
    
    try:
        # Try to get info for a specific Civitai model (if available)
        print("\nTrying to get model info (requires internet)...")
        model_info = civitai_client.get_model_info(25694)  # Popular anime model
        if model_info:
            print(f"✅ Model: {model_info.get('name', 'Unknown')}")
            print(f"   Type: {model_info.get('type', 'Unknown')}")
            print(f"   Description: {model_info.get('description', 'No description')[:100]}...")
            print(f"   Memory requirement: {model_info.get('memory_requirement', 'Unknown')}")
            
            versions = model_info.get('modelVersions', [])
            if versions:
                print(f"   Available versions: {len(versions)}")
                latest = versions[0]  # First is latest
                print(f"   Latest: {latest.get('name', 'Unknown')}")
        else:
            print("❌ Could not get model information (check internet/API key)")
    
    except Exception as e:
        print(f"❌ Model info unavailable: {e}")
    
    # Example 4: Download a model (demo - won't actually download)
    print("\n4. 📥 Model download example...")
    print("To download a Civitai model:")
    print("  1. Find the model ID from the URL or search results")
    print("  2. Use the model ID with 'civitai:' prefix")
    print("  3. Example: generator.download_model('My Model', 'civitai:12345')")
    print()
    print("Note: Downloads require storage space and may take time")
    print("Some models require Civitai API key for access")
    
    # Example 5: Show featured/popular models
    print("\n5. 🌟 Featured Civitai models...")
    try:
        featured = civitai_client.get_featured_models(content_type="image")
        if featured:
            print(f"Top {len(featured)} featured image models:")
            for model in featured[:3]:  # Show top 3
                print(f"  ⭐ {model.get('name', 'Unknown')}")
                print(f"     📥 {model.get('downloads', 0):,} downloads")
                print(f"     💾 {model.get('memory_requirement', 'Unknown')} memory")
                if model.get('nsfw'):
                    print(f"     🔞 NSFW content")
        else:
            print("No featured models found")
    
    except Exception as e:
        print(f"Error getting featured models: {e}")
    
    # Usage tips
    print("\n💡 Tips for using Civitai models:")
    print("  • Set CIVITAI_API_KEY for private/gated models")
    print("  • Check NSFW flags before using in production")
    print("  • Consider memory requirements for your hardware")
    print("  • Read model descriptions for usage instructions")
    print("  • Respect model licenses and creator rights")
    print("\n🔗 Useful links:")
    print("  • Civitai Models: https://civitai.com/models")
    print("  • API Documentation: https://developer.civitai.com")
    print("  • Your API Keys: https://civitai.com/user/account")


if __name__ == "__main__":
    main() 