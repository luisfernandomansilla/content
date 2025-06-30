#!/usr/bin/env python3
"""
Debug script to test model search functionality
"""
import os
import sys
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

# Add src to path
sys.path.insert(0, 'src')

def test_model_search():
    """Test the model search functionality step by step"""
    
    print("🔍 Debugging Model Search System")
    print("=" * 50)
    
    try:
        # Import components
        from content_creator import VideoGenerator
        from content_creator.models.model_manager import model_manager
        from content_creator.models.huggingface_client import HuggingFaceClient
        from content_creator.models.civitai_client import CivitaiClient
        
        print("✅ Imports successful")
        
        # Test 1: Check HuggingFace client
        print("\n🤗 Testing HuggingFace Client")
        print("-" * 30)
        
        hf_client = HuggingFaceClient()
        print(f"API Key present: {bool(hf_client.api_key)}")
        if hf_client.api_key:
            print(f"API Key preview: {hf_client.api_key[:10]}***")
        
        # Test HF search
        print("Testing HF video model search...")
        hf_results = hf_client.search_video_models("stable", limit=5)
        print(f"HF results count: {len(hf_results)}")
        
        for i, model in enumerate(hf_results[:3]):
            print(f"  {i+1}. {model.model_name} ({model.downloads} downloads)")
        
        # Test 2: Check Civitai client
        print("\n🎨 Testing Civitai Client")
        print("-" * 30)
        
        civitai_client = CivitaiClient()
        print(f"Civitai API key present: {bool(civitai_client.api_key)}")
        
        # Test Civitai search
        print("Testing Civitai model search...")
        civitai_results = civitai_client.search_models("anime", limit=5)
        print(f"Civitai results count: {len(civitai_results)}")
        
        for i, model in enumerate(civitai_results[:3]):
            print(f"  {i+1}. {model.get('name', 'Unknown')} (ID: {model.get('id', 'N/A')})")
        
        # Test 3: Check model manager
        print("\n📋 Testing Model Manager")
        print("-" * 30)
        
        # Test default models search
        default_results = model_manager.search_models("anime", platforms=["default"], limit=5)
        print(f"Default model results count: {len(default_results)}")
        
        for i, model in enumerate(default_results[:3]):
            print(f"  {i+1}. {model.get('name', 'Unknown')} (source: {model.get('source', 'N/A')})")
        
        # Test all platforms search
        all_results = model_manager.search_models("stable", platforms=["default", "huggingface", "civitai"], limit=10)
        print(f"All platforms results count: {len(all_results)}")
        
        for i, model in enumerate(all_results[:5]):
            print(f"  {i+1}. {model.get('name', 'Unknown')} (source: {model.get('source', 'N/A')})")
        
        # Test 4: Check VideoGenerator search
        print("\n🎬 Testing VideoGenerator Search")
        print("-" * 30)
        
        generator = VideoGenerator()
        video_results = generator.search_models("diffusion", platforms=["default", "huggingface"], limit=5)
        print(f"VideoGenerator results count: {len(video_results)}")
        
        for i, model in enumerate(video_results[:3]):
            print(f"  {i+1}. {model.get('name', 'Unknown')} (source: {model.get('source', 'N/A')})")
        
        # Test 5: Check main.py search function
        print("\n🔗 Testing Main Search Function")
        print("-" * 30)
        
        import main
        main_results, status = main.search_models("video", platform="all", limit=5)
        print(f"Main search results count: {len(main_results)}")
        print(f"Status: {status}")
        
        for i, (result_text, name) in enumerate(main_results[:3]):
            print(f"  {i+1}. {name}")
            print(f"     Preview: {result_text[:100]}...")
        
        print("\n📊 Summary")
        print("-" * 20)
        print(f"HF results: {len(hf_results)}")
        print(f"Civitai results: {len(civitai_results)}")
        print(f"Default results: {len(default_results)}")
        print(f"All platforms: {len(all_results)}")
        print(f"VideoGenerator: {len(video_results)}")
        print(f"Main function: {len(main_results)}")
        
        # Determine if working
        total_results = len(hf_results) + len(civitai_results) + len(default_results)
        
        if total_results > 0:
            print("✅ Search system is working!")
        else:
            print("❌ Search system has issues")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_search_with_different_queries():
    """Test search with various queries"""
    
    print("\n🔍 Testing Different Search Queries")
    print("=" * 50)
    
    from content_creator.models.model_manager import model_manager
    
    test_queries = [
        "stable",
        "diffusion", 
        "video",
        "anime",
        "flux",
        "text2video",
        ""  # Empty query
    ]
    
    for query in test_queries:
        try:
            results = model_manager.search_models(query, limit=3)
            print(f"Query: '{query}' -> {len(results)} results")
            
            if results:
                for i, model in enumerate(results[:2]):
                    print(f"  • {model.get('name', 'Unknown')} ({model.get('source', 'N/A')})")
        except Exception as e:
            print(f"Query: '{query}' -> Error: {e}")
        
        print()

if __name__ == "__main__":
    test_model_search()
    test_search_with_different_queries() 