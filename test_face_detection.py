#!/usr/bin/env python3

import cv2
import numpy as np
from PIL import Image
import sys

def test_face_detection(image_path):
    """Test face detection on an image"""
    try:
        # Load image using PIL first
        pil_image = Image.open(image_path)
        print(f"Image loaded: {pil_image.size}, mode: {pil_image.mode}")
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(pil_image)
        print(f"Image array shape: {image.shape}")
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV's pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces with different parameters
        print("Testing face detection with different parameters...")
        
        # Test 1: Default parameters
        faces1 = face_cascade.detectMultiScale(gray, 1.1, 4)
        print(f"Test 1 (1.1, 4): Found {len(faces1)} faces")
        
        # Test 2: More sensitive
        faces2 = face_cascade.detectMultiScale(gray, 1.05, 3)
        print(f"Test 2 (1.05, 3): Found {len(faces2)} faces")
        
        # Test 3: Even more sensitive
        faces3 = face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(30, 30))
        print(f"Test 3 (1.1, 2, minSize=30): Found {len(faces3)} faces")
        
        # Test 4: Very sensitive
        faces4 = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        print(f"Test 4 (1.3, 5, minSize=50): Found {len(faces4)} faces")
        
        # Use the best result
        all_faces = [faces1, faces2, faces3, faces4]
        best_faces = max(all_faces, key=len)
        
        if len(best_faces) > 0:
            print(f"\nBest result: {len(best_faces)} faces found")
            for i, (x, y, w, h) in enumerate(best_faces):
                print(f"Face {i+1}: x={x}, y={y}, w={w}, h={h}")
                
                # Add padding
                padding = int(min(w, h) * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                print(f"With padding: ({x1}, {y1}, {x2}, {y2})")
        else:
            print("\nNo faces detected! Using center crop fallback...")
            h, w = image.shape[:2]
            size = min(h, w)
            x1 = (w - size) // 2
            y1 = (h - size) // 2
            x2 = x1 + size
            y2 = y1 + size
            print(f"Center crop: ({x1}, {y1}, {x2}, {y2})")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_face_detection.py <image_path>")
        sys.exit(1)
    
    test_face_detection(sys.argv[1])
