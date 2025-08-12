#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.musetalk_model import MuseTalkModel
import numpy as np
from PIL import Image

def test_musetalk():
    """Test MuseTalk model with a real image"""
    try:
        # Initialize MuseTalk model
        print("Initializing MuseTalk model...")
        model = MuseTalkModel(device='cpu')
        
        # Load a test image
        image_path = "outputs/input_image_d7280166-f71d-4ec3-9af4-7916c4a64025.jpg"
        print(f"Loading image: {image_path}")
        
        # Load image using PIL
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(pil_image)
        print(f"Image shape: {image.shape}")
        
        # Test face detection
        print("Testing face detection...")
        bbox = model._detect_face(image)
        print(f"Face bbox: {bbox}")
        
        # Test face feature extraction
        print("Testing face feature extraction...")
        face_features = model._extract_face_features(image, bbox)
        print(f"Face features shape: {face_features.shape}")
        
        print("✅ MuseTalk model test completed successfully!")
        
    except Exception as e:
        print(f"❌ MuseTalk model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_musetalk()
