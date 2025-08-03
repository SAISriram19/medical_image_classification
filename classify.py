#!/usr/bin/env python3
"""
Medical Image Classifier - Command Line Interface

A production-ready CLI for classifying images as medical or non-medical
using advanced AI ensemble methods.

Author: Medical AI Team
License: MIT
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

import argparse
import time
from typing import List, Tuple, Dict
import json

from src.image_extractor import ImageExtractor
from src.classifier import MedicalImageClassifier
from src.utils import setup_logging, save_results

def main():
    parser = argparse.ArgumentParser(description='Classify images as medical or non-medical')
    parser.add_argument('--url', type=str, help='Website URL to extract images from')
    parser.add_argument('--pdf', type=str, help='PDF file path to extract images from')
    parser.add_argument('--output', type=str, default='results.json', help='Output file for results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if not args.url and not args.pdf:
        parser.error("Either --url or --pdf must be provided")
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Initialize components
    extractor = ImageExtractor()
    classifier = MedicalImageClassifier()
    
    start_time = time.time()
    
    try:
        # Extract images
        if args.url:
            logger.info(f"Extracting images from URL: {args.url}")
            images = extractor.extract_from_url(args.url)
        else:
            logger.info(f"Extracting images from PDF: {args.pdf}")
            images = extractor.extract_from_pdf(args.pdf)
        
        logger.info(f"Extracted {len(images)} images")
        
        # Classify images
        results = []
        for i, (image, source) in enumerate(images):
            prediction, confidence = classifier.predict(image)
            results.append({
                'image_id': i,
                'source': source,
                'prediction': prediction,
                'confidence': float(confidence)
            })
            logger.info(f"Image {i}: {prediction} (confidence: {confidence:.3f})")
        
        # Save results
        processing_time = time.time() - start_time
        output_data = {
            'input_source': args.url or args.pdf,
            'total_images': len(images),
            'processing_time_seconds': processing_time,
            'results': results
        }
        
        save_results(output_data, args.output)
        
        # Summary
        medical_count = sum(1 for r in results if r['prediction'] == 'medical')
        non_medical_count = len(results) - medical_count
        
        print(f"\n=== Classification Results ===")
        print(f"Total images processed: {len(results)}")
        print(f"Medical images: {medical_count}")
        print(f"Non-medical images: {non_medical_count}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Average time per image: {processing_time/len(results):.3f} seconds")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()