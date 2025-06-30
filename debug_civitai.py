#!/usr/bin/env python3
"""
Debug script to inspect Civitai API response
"""
import os
import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

# Add src to path
sys.path.insert(0, 'src')

def inspect_civitai_response():
    """Inspect the raw Civitai API response to understand the parsing issue"""
    
    from content_creator.models.civitai_client import CivitaiClient
    
    print("🎨 Debugging Civitai API Response")
    print("=" * 50)
    
    client = CivitaiClient()
    
    try:
        # Make a direct API call to inspect the response
        response = client.session.get(f"{client.base_url}/v1/models", params={
            "query": "stable",
            "limit": 2,
            "sort": "Highest Rated",
            "period": "AllTime"
        })
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print()
        
        data = response.json()
        print(f"Response keys: {list(data.keys())}")
        print(f"Items count: {len(data.get('items', []))}")
        print()
        
        # Inspect each item
        for i, item in enumerate(data.get("items", [])[:2]):
            print(f"Item {i+1}:")
            print(f"  Type: {type(item)}")
            print(f"  Keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}")
            print()
            
            if isinstance(item, dict):
                # Check specific fields that might be problematic
                print(f"  id: {item.get('id')} (type: {type(item.get('id'))})")
                print(f"  name: {item.get('name')} (type: {type(item.get('name'))})")
                print(f"  description: {item.get('description')} (type: {type(item.get('description'))})")
                print(f"  type: {item.get('type')} (type: {type(item.get('type'))})")
                print(f"  creator: {item.get('creator')} (type: {type(item.get('creator'))})")
                print(f"  tags: {item.get('tags')} (type: {type(item.get('tags'))})")
                print(f"  stats: {item.get('stats')} (type: {type(item.get('stats'))})")
                print(f"  modelVersions: {item.get('modelVersions')} (type: {type(item.get('modelVersions'))})")
                print()
                
                # Test the parsing function
                print(f"  Testing _parse_model_info on this item...")
                try:
                    parsed = client._parse_model_info(item)
                    print(f"  ✅ Parsing successful: {parsed.get('name', 'Unknown') if parsed else 'None'}")
                except Exception as e:
                    print(f"  ❌ Parsing failed: {e}")
                    print(f"  Error type: {type(e)}")
                    
                    # Let's see which specific field is causing the issue
                    problematic_fields = []
                    
                    for field in ['creator', 'tags', 'stats', 'modelVersions']:
                        field_value = item.get(field)
                        if field_value is not None:
                            print(f"    {field}: {field_value} (type: {type(field_value)})")
                            
                            # Check if it's a list and what's inside
                            if isinstance(field_value, list) and field_value:
                                print(f"      First element type: {type(field_value[0])}")
                                if isinstance(field_value[0], dict):
                                    print(f"      First element keys: {list(field_value[0].keys())}")
                                else:
                                    print(f"      First element value: {field_value[0]}")
                            
                            # Check if it's a dict
                            elif isinstance(field_value, dict):
                                print(f"      Dict keys: {list(field_value.keys())}")
                            
                            # Check if it's a string (might be the issue)
                            elif isinstance(field_value, str):
                                print(f"      String value: '{field_value[:100]}...'")
                                problematic_fields.append(field)
                    
                    if problematic_fields:
                        print(f"    Potential problematic fields (strings): {problematic_fields}")
                
                print("-" * 30)
        
        # Show the metadata too
        if 'metadata' in data:
            print(f"Metadata: {data['metadata']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_parse_model_info_directly():
    """Test the _parse_model_info function with manual data"""
    
    print("\n🔧 Testing _parse_model_info directly")
    print("=" * 50)
    
    from content_creator.models.civitai_client import CivitaiClient
    
    client = CivitaiClient()
    
    # Create test data that mimics what we might get from the API
    test_data = {
        "id": 12345,
        "name": "Test Model",
        "description": "A test model",
        "type": "Checkpoint",
        "creator": {"username": "testuser"}, # This should be a dict
        "tags": [{"name": "anime"}, {"name": "realistic"}], # This should be a list of dicts
        "stats": {"rating": 4.5, "downloadCount": 1000}, # This should be a dict
        "modelVersions": [{"id": 1, "name": "v1.0"}], # This should be a list of dicts
        "nsfw": False,
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z"
    }
    
    print("Testing with well-formed data:")
    try:
        result = client._parse_model_info(test_data)
        print(f"✅ Success: {result.get('name') if result else 'None'}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Now test with problematic data (strings instead of dicts)
    test_data_bad = test_data.copy()
    test_data_bad["creator"] = "testuser"  # String instead of dict
    test_data_bad["tags"] = ["anime", "realistic"]  # List of strings instead of list of dicts
    
    print("\nTesting with potentially problematic data:")
    try:
        result = client._parse_model_info(test_data_bad)
        print(f"✅ Success: {result.get('name') if result else 'None'}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        print(f"Error details: {type(e).__name__}: {e}")

if __name__ == "__main__":
    inspect_civitai_response()
    test_parse_model_info_directly() 