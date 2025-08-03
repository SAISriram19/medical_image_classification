# Medical vs Non-Medical Image Classification Approach

## Overview
This project implements a machine learning pipeline to classify images as medical or non-medical using a combination of heuristic analysis and computer vision techniques.

## Architecture

### 1. Image Extraction (`src/image_extractor.py`)
- **URL Extraction**: Uses BeautifulSoup to parse HTML and extract image URLs
- **PDF Extraction**: Converts PDF pages to images using pdf2image
- **Filtering**: Removes small images (likely icons/logos) with minimum size threshold
- **Preprocessing**: Ensures all images are in RGB format

### 2. Classification (`src/classifier.py`)
- **Primary Method**: Heuristic-based analysis optimized for medical image characteristics
- **Backup**: Pre-trained ResNet-50 model for additional validation
- **Features Analyzed**:
  - Contrast levels (medical images often have high contrast)
  - Color distribution (medical images often grayscale or limited palette)
  - Edge density (medical images have clear anatomical structures)
  - Intensity distribution patterns (bimodal distributions common in medical imaging)

### 3. Processing Pipeline (`classify.py`)
- Command-line interface supporting both URL and PDF inputs
- Batch processing with progress tracking
- JSON output with detailed results and metadata

## Technical Approach

### Heuristic Analysis
The classifier uses several visual characteristics typical of medical images:

1. **Contrast Analysis**: Medical images (X-rays, CT scans) typically have high contrast
2. **Grayscale Tendency**: Many medical images are grayscale or have limited color palettes
3. **Edge Density**: Anatomical structures create distinct edges and boundaries
4. **Intensity Distribution**: Medical images often show bimodal intensity distributions

### Model Selection Rationale
- **Open Source**: Uses transformers library and standard computer vision techniques
- **Efficiency**: Heuristic approach provides fast inference without GPU requirements
- **Scalability**: Lightweight processing suitable for batch operations
- **Accuracy**: Combines multiple visual features for robust classification

## Performance Considerations

### Speed Optimizations
- Image resizing for large inputs (max 512x512)
- Efficient numpy operations for image analysis
- Minimal model loading overhead
- Batch processing capabilities

### Memory Efficiency
- Processes images individually to avoid memory buildup
- Automatic garbage collection between images
- Configurable batch sizes for large datasets

### Scalability Features
- Stateless processing (can be easily parallelized)
- JSON output format for easy integration
- Modular design for component replacement
- Error handling and recovery

## Validation Strategy

### Test Cases
- Synthetic medical images (high contrast, grayscale structures)
- Synthetic non-medical images (colorful, natural patterns)
- Real-world validation on publicly available medical datasets

### Metrics
- Classification accuracy on validation set
- Processing time per image
- Memory usage during batch processing
- False positive/negative analysis

## Future Improvements

1. **Enhanced Training**: Fine-tune on medical image datasets
2. **Multi-class Classification**: Distinguish between X-ray, MRI, CT scan types
3. **Confidence Calibration**: Improve confidence score reliability
4. **GPU Acceleration**: Optional CUDA support for faster processing
5. **Advanced Features**: DICOM format support, metadata analysis

## Dependencies
- Core ML: PyTorch, Transformers, NumPy, SciPy
- Image Processing: Pillow, pdf2image
- Web Scraping: Requests, BeautifulSoup4
- Utilities: Standard Python libraries

This approach balances accuracy, speed, and simplicity while maintaining the flexibility to incorporate more sophisticated models as needed.