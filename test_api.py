#!/usr/bin/env python3

import requests
import json

def test_generate_video():
    """Test the video generation API"""
    url = "http://localhost:8000/api/generate"
    
    # Test data with short text for debugging
    data = {
        "text": "Hello world test",
        "voice": "v2/en_speaker_1"
    }
    
    # Test image file
    image_path = "outputs/input_image_d7280166-f71d-4ec3-9af4-7916c4a64025.jpg"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': (image_path, f, 'image/jpeg')}

            print("Sending request to generate video...")
            response = requests.post(url, data=data, files=files, timeout=300)
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success! Response: {result}")
            else:
                print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_generate_video()
