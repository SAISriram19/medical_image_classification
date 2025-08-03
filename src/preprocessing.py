"""
Advanced image preprocessing for robust medical image classification
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class AdvancedImagePreprocessor:
    """Advanced preprocessing pipeline for medical images"""
    
    def __init__(self):
        self.target_size = (512, 512)
        self.enhancement_params = {
            'contrast_range': (0.8, 1.5),
            'brightness_range': (0.9, 1.1),
            'sharpness_range': (0.8, 1.2)
        }
    
    def preprocess_image(self, image: Image.Image, enhance: bool = True) -> Image.Image:
        """
        Comprehensive preprocessing pipeline
        """
        try:
            # 1. Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 2. Resize while maintaining aspect ratio
            image = self._smart_resize(image)
            
            # 3. Noise reduction
            image = self._denoise_image(image)
            
            # 4. Contrast enhancement (optional)
            if enhance:
                image = self._enhance_image(image)
            
            # 5. Normalization
            image = self._normalize_image(image)
            
            return image
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return image
    
    def _smart_resize(self, image: Image.Image) -> Image.Image:
        """Resize image while maintaining aspect ratio and padding if needed"""
        original_size = image.size
        target_w, target_h = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / original_size[0], target_h / original_size[1])
        
        # Resize
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Pad to target size
        if new_size != self.target_size:
            # Create new image with padding
            padded = Image.new('RGB', self.target_size, (0, 0, 0))
            
            # Calculate padding
            pad_x = (target_w - new_size[0]) // 2
            pad_y = (target_h - new_size[1]) // 2
            
            # Paste resized image
            padded.paste(image, (pad_x, pad_y))
            image = padded
        
        return image
    
    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """Apply noise reduction techniques"""
        # Convert to numpy for OpenCV operations
        img_array = np.array(image)
        
        # Apply bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # Convert back to PIL
        return Image.fromarray(denoised)
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply adaptive enhancement based on image characteristics"""
        # Analyze image characteristics
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Calculate image statistics
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # Adaptive enhancement based on image characteristics
        if mean_intensity < 100:  # Dark image (possibly X-ray)
            # Increase contrast and brightness
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
        elif std_intensity < 30:  # Low contrast image
            # Increase contrast significantly
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
        else:  # Normal image
            # Mild enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
        
        # Adaptive sharpening
        if std_intensity > 50:  # High detail image
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
        
        return image
    
    def _normalize_image(self, image: Image.Image) -> Image.Image:
        """Normalize image intensity values"""
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize each channel
        for channel in range(img_array.shape[2]):
            channel_data = img_array[:, :, channel]
            
            # Robust normalization using percentiles
            p2, p98 = np.percentile(channel_data, (2, 98))
            
            if p98 > p2:
                channel_data = np.clip((channel_data - p2) / (p98 - p2) * 255, 0, 255)
                img_array[:, :, channel] = channel_data
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def create_augmented_versions(self, image: Image.Image, num_versions: int = 3) -> List[Image.Image]:
        """Create multiple augmented versions for ensemble prediction"""
        versions = [image]  # Original
        
        for i in range(num_versions - 1):
            augmented = image.copy()
            
            # Random rotation (small angles)
            angle = np.random.uniform(-5, 5)
            augmented = augmented.rotate(angle, fillcolor=(0, 0, 0))
            
            # Random contrast adjustment
            contrast_factor = np.random.uniform(0.9, 1.1)
            enhancer = ImageEnhance.Contrast(augmented)
            augmented = enhancer.enhance(contrast_factor)
            
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.95, 1.05)
            enhancer = ImageEnhance.Brightness(augmented)
            augmented = enhancer.enhance(brightness_factor)
            
            versions.append(augmented)
        
        return versions
    
    def extract_roi(self, image: Image.Image) -> Image.Image:
        """Extract region of interest using edge detection"""
        try:
            # Convert to grayscale
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.width - x, w + 2 * padding)
                h = min(image.height - y, h + 2 * padding)
                
                # Crop image
                roi = image.crop((x, y, x + w, y + h))
                
                # Resize back to standard size
                roi = roi.resize(self.target_size, Image.Resampling.LANCZOS)
                
                return roi
            
            return image
            
        except Exception as e:
            logger.warning(f"ROI extraction failed: {e}")
            return image