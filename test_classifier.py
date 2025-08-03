#!/usr/bin/env python3
"""
Medical Image Classifier - Test Suite

Basic functionality tests for the medical image classification system.

Author: Medical AI Team
License: MIT
"""

import sys
import os
from pathlib import Path
from PIL import Image

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from src.classifier import MedicalImageClassifier
from src.utils import setup_logging

def main():
    """Test the classifier initialization and basic functionality"""
    logger = setup_logging(verbose=True)
    
    print("Testing Medical Image Classifier...")
    
    try:
        # Initialize classifier
        classifier = MedicalImageClassifier()
        print("✓ Classifier initialized successfully")
        
        # Test with a simple image
        test_image = Image.new('RGB', (256, 256), color=(128, 128, 128))
        
        # Test prediction
        prediction, confidence = classifier.predict(test_image)
        print(f"✓ Prediction test: {prediction} (confidence: {confidence:.3f})")
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()