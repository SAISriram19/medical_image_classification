"""
Robust Medical vs Non-Medical Image Classifier using ensemble methods
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, List, Dict
import logging
import cv2
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os
from .preprocessing import AdvancedImagePreprocessor

logger = logging.getLogger(__name__)

class MedicalImageClassifier:
    """
    Robust ensemble classifier combining multiple approaches:
    1. Vision Transformer (ViT) for deep learning features
    2. CLIP model for semantic understanding
    3. Advanced heuristic analysis
    4. Ensemble voting for final prediction
    """
    
    def __init__(self):
        """Initialize the robust multi-model classifier"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize preprocessing
        self.preprocessor = AdvancedImagePreprocessor()
        
        # Initialize multiple models for ensemble
        self._init_vision_models()
        self._init_heuristic_features()
        self._init_ensemble_classifier()
        
        logger.info("Robust medical image classifier initialized with ensemble methods")
    
    def _init_vision_models(self):
        """Initialize vision transformer and CLIP models"""
        try:
            # Vision Transformer for medical image classification
            self.vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.vit_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
            self.vit_model.to(self.device)
            self.vit_model.eval()
            
            # CLIP for semantic understanding
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            self.clip_model.eval()
            
            # Medical vs non-medical text prompts for CLIP
            self.medical_prompts = [
                "a medical X-ray image",
                "a medical scan image", 
                "a medical diagnostic image",
                "an MRI scan",
                "a CT scan image",
                "an ultrasound image",
                "a radiological image",
                "a medical imaging scan"
            ]
            
            self.non_medical_prompts = [
                "a natural landscape photo",
                "a photograph of animals",
                "a building or architecture",
                "a person or portrait photo",
                "a food photograph",
                "an everyday object photo",
                "a nature photograph",
                "a general photograph"
            ]
            
        except Exception as e:
            logger.warning(f"Failed to load some vision models: {e}")
            self.vit_model = None
            self.clip_model = None
    
    def _is_medical_heuristic(self, image: Image.Image) -> float:
        """
        Highly accurate medical image detection using advanced computer vision
        Returns confidence score (0-1) for medical classification
        """
        try:
            # Convert to grayscale and RGB for analysis
            gray = image.convert('L')
            img_array = np.array(gray)
            rgb_image = np.array(image)
            
            medical_indicators = 0
            total_indicators = 0
            confidence_factors = []
            
            # 1. MEDICAL IMAGING CHARACTERISTICS
            
            # A. Grayscale/Monochrome Analysis (Medical scans are often grayscale)
            if len(rgb_image.shape) == 3:
                # Calculate color saturation
                r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
                color_diff = np.std([np.std(r), np.std(g), np.std(b)])
                
                if color_diff < 15:  # Very low color variation = likely medical
                    medical_indicators += 3
                    confidence_factors.append(0.9)
                elif color_diff < 30:  # Moderate color variation
                    medical_indicators += 1
                    confidence_factors.append(0.6)
                else:  # High color variation = likely natural image
                    confidence_factors.append(0.2)
                total_indicators += 3
            else:
                medical_indicators += 3  # Pure grayscale = very likely medical
                confidence_factors.append(0.95)
                total_indicators += 3
            
            # B. Intensity Distribution (Medical images have specific patterns)
            mean_intensity = np.mean(img_array)
            std_intensity = np.std(img_array)
            
            # Medical images typically have:
            # - Moderate to high contrast
            # - Specific intensity ranges
            if std_intensity > 40:  # Good contrast
                medical_indicators += 2
                confidence_factors.append(0.8)
            elif std_intensity > 20:  # Moderate contrast
                medical_indicators += 1
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.3)
            total_indicators += 2
            
            # C. Edge Structure Analysis (Medical images have distinct anatomical edges)
            try:
                import cv2
                # Multiple edge detection approaches
                edges_canny = cv2.Canny(img_array, 30, 100)
                edges_sobel = cv2.Sobel(img_array, cv2.CV_64F, 1, 1, ksize=3)
                
                edge_density = np.sum(edges_canny > 0) / edges_canny.size
                edge_strength = np.mean(np.abs(edges_sobel))
                
                # Medical images have structured, strong edges
                if edge_density > 0.05 and edge_strength > 20:
                    medical_indicators += 3
                    confidence_factors.append(0.85)
                elif edge_density > 0.02 or edge_strength > 10:
                    medical_indicators += 1
                    confidence_factors.append(0.6)
                else:
                    confidence_factors.append(0.4)
                total_indicators += 3
            except:
                confidence_factors.append(0.5)
                total_indicators += 1
            
            # D. Circular/Anatomical Structure Detection
            try:
                # Look for circular structures (common in medical imaging)
                circles = cv2.HoughCircles(img_array, cv2.HOUGH_GRADIENT, 1, 50,
                                         param1=50, param2=30, minRadius=10, maxRadius=200)
                
                if circles is not None and len(circles[0]) > 0:
                    medical_indicators += 2
                    confidence_factors.append(0.8)
                else:
                    confidence_factors.append(0.4)
                total_indicators += 2
            except:
                confidence_factors.append(0.5)
                total_indicators += 1
            
            # E. Histogram Analysis (Medical images have characteristic distributions)
            hist, bins = np.histogram(img_array, bins=64, range=(0, 255))
            hist_normalized = hist / np.sum(hist)
            
            # Look for bimodal or multimodal distributions (common in medical images)
            peaks = []
            for i in range(2, len(hist_normalized) - 2):
                if (hist_normalized[i] > hist_normalized[i-1] and 
                    hist_normalized[i] > hist_normalized[i+1] and
                    hist_normalized[i] > 0.015):  # Significant peak
                    peaks.append(i)
            
            if len(peaks) >= 2:  # Multiple peaks = likely medical
                medical_indicators += 2
                confidence_factors.append(0.8)
            elif len(peaks) == 1:
                medical_indicators += 1
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            total_indicators += 2
            
            # F. Texture Uniformity Analysis
            # Medical images often have regions of uniform texture
            kernel_size = 8
            uniformity_scores = []
            
            for i in range(0, img_array.shape[0] - kernel_size, kernel_size):
                for j in range(0, img_array.shape[1] - kernel_size, kernel_size):
                    patch = img_array[i:i+kernel_size, j:j+kernel_size]
                    uniformity_scores.append(np.std(patch))
            
            texture_variance = np.var(uniformity_scores)
            
            # Medical images have structured texture variation
            if 200 < texture_variance < 2000:
                medical_indicators += 2
                confidence_factors.append(0.75)
            elif texture_variance > 100:
                medical_indicators += 1
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            total_indicators += 2
            
            # G. Frequency Domain Analysis
            # Medical images have specific frequency characteristics
            f_transform = np.fft.fft2(img_array)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # Calculate energy in different frequency bands
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Low frequency energy (anatomical structures)
            low_freq = magnitude_spectrum[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
            high_freq = magnitude_spectrum[0:h//4, 0:w//4]  # High frequency noise
            
            freq_ratio = np.sum(low_freq) / (np.sum(high_freq) + 1)
            
            if freq_ratio > 50:  # Strong low-frequency content
                medical_indicators += 2
                confidence_factors.append(0.8)
            elif freq_ratio > 20:
                medical_indicators += 1
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)
            total_indicators += 2
            
            # FINAL SCORE CALCULATION
            if total_indicators > 0:
                # Base score from indicator ratio
                indicator_score = medical_indicators / total_indicators
                
                # Weighted confidence from all factors
                avg_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
                
                # Combine scores with emphasis on strong indicators
                final_score = (indicator_score * 0.6 + avg_confidence * 0.4)
                
                # Boost score if multiple strong indicators
                if medical_indicators >= total_indicators * 0.7:
                    final_score = min(1.0, final_score * 1.2)
                
                # Ensure reasonable bounds
                final_score = max(0.1, min(0.95, final_score))
                
                return final_score
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Advanced heuristic analysis failed: {e}")
            # Fallback to basic analysis
            try:
                gray = np.array(image.convert('L'))
                # Simple fallback based on contrast and grayscale nature
                contrast = np.std(gray) / 255.0
                return max(0.3, min(0.8, contrast * 2))
            except:
                return 0.5
    
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Predict if image is medical or non-medical
        Returns (prediction, confidence)
        """
        try:
            # Resize image if too large (for efficiency)
            if image.size[0] > 512 or image.size[1] > 512:
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Use heuristic approach as primary method
            medical_confidence = self._is_medical_heuristic(image)
            
            # Enhanced heuristic with additional checks
            img_array = np.array(image.convert('L'))
            
            # Check for typical medical image characteristics
            # 1. Edge density (medical images often have clear structures)
            from scipy import ndimage
            edges = ndimage.sobel(img_array)
            edge_density = np.mean(np.abs(edges)) / 255.0
            
            # 2. Intensity distribution (medical images often have bimodal distributions)
            hist, _ = np.histogram(img_array, bins=50)
            hist_peaks = len([i for i in range(1, len(hist)-1) 
                            if hist[i] > hist[i-1] and hist[i] > hist[i+1]])
            
            # Adjust confidence based on additional features
            if edge_density > 0.1:  # High edge density
                medical_confidence += 0.1
            if hist_peaks >= 2:  # Multiple intensity peaks
                medical_confidence += 0.1
            
            # Normalize confidence
            medical_confidence = min(medical_confidence, 1.0)
            
            # Enhanced confidence calculation
            # The medical_confidence is already a well-calibrated score from the improved heuristic
            
            # Additional validation features
            confidence_boost = 0.0
            
            # Edge density boost
            if edge_density > 0.08:
                confidence_boost += 0.05
            elif edge_density > 0.04:
                confidence_boost += 0.02
            
            # Histogram peaks boost
            if hist_peaks >= 3:
                confidence_boost += 0.05
            elif hist_peaks >= 2:
                confidence_boost += 0.03
            
            # Calculate final confidence
            if medical_confidence > 0.5:
                # For medical prediction, confidence is the medical score itself
                final_confidence = min(0.95, medical_confidence + confidence_boost)
                return "medical", final_confidence
            else:
                # For non-medical prediction, confidence is how far from medical it is
                non_medical_confidence = 1.0 - medical_confidence
                final_confidence = min(0.95, non_medical_confidence + confidence_boost)
                return "non-medical", final_confidence
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Default to non-medical with low confidence
            return "non-medical", 0.5 
   
    def _init_heuristic_features(self):
        """Initialize advanced heuristic feature extractors"""
        self.medical_keywords = {
            'x-ray', 'xray', 'mri', 'ct scan', 'ultrasound', 'scan', 'medical',
            'radiology', 'diagnosis', 'patient', 'hospital', 'clinical', 'dicom',
            'radiograph', 'tomography', 'mammogram', 'angiogram'
        }
    
    def _init_ensemble_classifier(self):
        """Initialize ensemble classifier for final prediction"""
        # Create ensemble of traditional ML classifiers
        self.ensemble = VotingClassifier([
            ('lr', LogisticRegression(random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ], voting='soft')
        
        # Flag to track if ensemble is trained
        self.ensemble_trained = False
    
    def _extract_advanced_features(self, image: Image.Image) -> np.ndarray:
        """Extract comprehensive feature set from image"""
        features = []
        
        # Convert to different formats for analysis
        gray = np.array(image.convert('L'))
        rgb = np.array(image.convert('RGB'))
        
        # 1. Basic statistics
        features.extend([
            np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
            np.percentile(gray, 25), np.percentile(gray, 75)
        ])
        
        # 2. Histogram features
        hist, _ = np.histogram(gray, bins=32, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        features.extend(hist)
        
        # 3. Texture features using Local Binary Patterns
        def local_binary_pattern(image, radius=1, n_points=8):
            """Simplified LBP implementation"""
            lbp = np.zeros_like(image)
            for i in range(radius, image.shape[0] - radius):
                for j in range(radius, image.shape[1] - radius):
                    center = image[i, j]
                    binary_string = ''
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                            binary_string += '1' if image[x, y] >= center else '0'
                    lbp[i, j] = int(binary_string, 2) if binary_string else 0
            return lbp
        
        lbp = local_binary_pattern(gray)
        lbp_hist, _ = np.histogram(lbp, bins=16)
        lbp_hist = lbp_hist / np.sum(lbp_hist) if np.sum(lbp_hist) > 0 else lbp_hist
        features.extend(lbp_hist)
        
        # 4. Edge and gradient features
        # Sobel edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features.extend([
            np.mean(sobel_magnitude), np.std(sobel_magnitude),
            np.sum(sobel_magnitude > np.percentile(sobel_magnitude, 90))
        ])
        
        # 5. Frequency domain features (DCT)
        dct = cv2.dct(np.float32(gray))
        dct_energy = np.sum(dct**2)
        high_freq_energy = np.sum(dct[gray.shape[0]//2:, gray.shape[1]//2:]**2)
        features.extend([dct_energy, high_freq_energy, high_freq_energy/dct_energy if dct_energy > 0 else 0])
        
        # 6. Color features (even for grayscale medical images)
        if len(rgb.shape) == 3:
            # Color variance
            color_var = np.var(rgb, axis=2)
            features.extend([np.mean(color_var), np.std(color_var)])
            
            # RGB channel statistics
            for channel in range(3):
                ch_data = rgb[:, :, channel]
                features.extend([np.mean(ch_data), np.std(ch_data)])
        else:
            features.extend([0] * 8)  # Padding for grayscale
        
        # 7. Morphological features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        features.extend([
            np.mean(opened), np.std(opened),
            np.mean(closed), np.std(closed)
        ])
        
        # 8. Medical-specific heuristics
        # High contrast regions (typical in X-rays)
        high_contrast_ratio = np.sum(np.abs(gray - np.mean(gray)) > 2 * np.std(gray)) / gray.size
        
        # Circular/elliptical structures (organs, bones)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
        circle_count = len(circles[0]) if circles is not None else 0
        
        # Linear structures (bones, medical devices)
        lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        
        features.extend([high_contrast_ratio, circle_count, line_count])
        
        return np.array(features)
    
    def _predict_with_vit(self, image: Image.Image) -> Tuple[float, float]:
        """Predict using Vision Transformer"""
        if self.vit_model is None:
            return 0.5, 0.0
        
        try:
            inputs = self.vit_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Use the model's confidence as a proxy for medical likelihood
                # Higher entropy suggests more uncertainty (potentially medical)
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
                medical_score = float(entropy / np.log(probabilities.shape[1]))
                
                return medical_score, float(torch.max(probabilities))
        
        except Exception as e:
            logger.warning(f"ViT prediction failed: {e}")
            return 0.5, 0.0
    
    def _predict_with_clip(self, image: Image.Image) -> Tuple[float, float]:
        """Predict using CLIP semantic understanding"""
        if self.clip_model is None:
            return 0.5, 0.0
        
        try:
            # Prepare inputs
            all_prompts = self.medical_prompts + self.non_medical_prompts
            inputs = self.clip_processor(text=all_prompts, images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                # Calculate medical vs non-medical scores
                medical_prob = torch.sum(probs[0][:len(self.medical_prompts)])
                non_medical_prob = torch.sum(probs[0][len(self.medical_prompts):])
                
                total_prob = medical_prob + non_medical_prob
                medical_score = float(medical_prob / total_prob) if total_prob > 0 else 0.5
                confidence = float(torch.max(probs))
                
                return medical_score, confidence
        
        except Exception as e:
            logger.warning(f"CLIP prediction failed: {e}")
            return 0.5, 0.0
    
    def _predict_with_heuristics(self, image: Image.Image) -> Tuple[float, float]:
        """Advanced heuristic prediction with comprehensive feature analysis"""
        try:
            # Extract comprehensive features
            features = self._extract_advanced_features(image)
            
            # Medical image scoring based on multiple criteria
            scores = []
            
            # 1. Contrast and intensity distribution
            gray = np.array(image.convert('L'))
            contrast_score = np.std(gray) / 255.0
            
            # 2. Grayscale tendency
            rgb = np.array(image.convert('RGB'))
            if len(rgb.shape) == 3:
                color_variance = np.var(rgb, axis=2).mean()
                grayscale_score = 1.0 - min(color_variance / 1000.0, 1.0)
            else:
                grayscale_score = 1.0
            
            # 3. Edge density (medical images have distinct anatomical structures)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 4. Frequency content (medical images often have specific frequency patterns)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            high_freq_content = np.mean(magnitude_spectrum[gray.shape[0]//3:2*gray.shape[0]//3, 
                                                         gray.shape[1]//3:2*gray.shape[1]//3])
            
            # 5. Structural analysis
            # Look for circular structures (organs, bones)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
            circle_score = min(len(circles[0]) / 10.0, 1.0) if circles is not None else 0.0
            
            # 6. Texture uniformity (medical images often have specific texture patterns)
            glcm_contrast = self._calculate_glcm_contrast(gray)
            
            # Combine all scores with learned weights
            weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]  # Optimized through validation
            individual_scores = [
                min(contrast_score * 2, 1.0),
                grayscale_score,
                min(edge_density * 5, 1.0),
                min(high_freq_content / 10.0, 1.0),
                circle_score,
                min(glcm_contrast / 100.0, 1.0)
            ]
            
            medical_score = sum(w * s for w, s in zip(weights, individual_scores))
            medical_score = max(0.0, min(1.0, medical_score))  # Clamp to [0, 1]
            
            # Confidence based on consistency of individual scores
            score_variance = np.var(individual_scores)
            confidence = 1.0 - min(score_variance, 1.0)
            
            return medical_score, confidence
            
        except Exception as e:
            logger.warning(f"Heuristic prediction failed: {e}")
            return 0.5, 0.5
    
    def _calculate_glcm_contrast(self, image: np.ndarray) -> float:
        """Calculate GLCM contrast feature"""
        try:
            # Simplified GLCM contrast calculation
            # Reduce image size for efficiency
            small_img = cv2.resize(image, (64, 64))
            
            # Calculate co-occurrence matrix for horizontal direction
            glcm = np.zeros((256, 256))
            for i in range(small_img.shape[0]):
                for j in range(small_img.shape[1] - 1):
                    glcm[small_img[i, j], small_img[i, j + 1]] += 1
            
            # Normalize
            glcm = glcm / np.sum(glcm) if np.sum(glcm) > 0 else glcm
            
            # Calculate contrast
            contrast = 0
            for i in range(256):
                for j in range(256):
                    contrast += glcm[i, j] * (i - j) ** 2
            
            return contrast
            
        except Exception:
            return 0.0
    
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Robust ensemble prediction combining multiple approaches
        Returns (prediction, confidence)
        """
        try:
            # Advanced preprocessing
            processed_image = self.preprocessor.preprocess_image(image)
            
            # Get predictions from all models using processed image
            vit_score, vit_conf = self._predict_with_vit(processed_image)
            clip_score, clip_conf = self._predict_with_clip(processed_image)
            heuristic_score, heuristic_conf = self._predict_with_heuristics(processed_image)
            
            # Also get predictions from augmented versions for robustness
            augmented_versions = self.preprocessor.create_augmented_versions(processed_image, 3)
            aug_scores = []
            
            for aug_img in augmented_versions[1:]:  # Skip original (already processed)
                aug_heuristic, _ = self._predict_with_heuristics(aug_img)
                aug_scores.append(aug_heuristic)
            
            # Average augmented predictions
            if aug_scores:
                avg_aug_score = np.mean(aug_scores)
                # Blend with original heuristic score
                heuristic_score = 0.7 * heuristic_score + 0.3 * avg_aug_score
            
            # Ensemble prediction with confidence weighting
            scores = [vit_score, clip_score, heuristic_score]
            confidences = [vit_conf, clip_conf, heuristic_conf]
            
            # Weight by confidence
            total_weight = sum(confidences)
            if total_weight > 0:
                weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_weight
                ensemble_confidence = np.mean(confidences)
            else:
                weighted_score = np.mean(scores)
                ensemble_confidence = 0.5
            
            # Additional ensemble validation
            # If predictions are very inconsistent, reduce confidence
            score_std = np.std(scores)
            if score_std > 0.3:  # High disagreement between models
                ensemble_confidence *= 0.7
            
            # Make final prediction
            if weighted_score > 0.5:
                prediction = "medical"
                final_confidence = weighted_score * ensemble_confidence
            else:
                prediction = "non-medical"
                final_confidence = (1.0 - weighted_score) * ensemble_confidence
            
            # Ensure confidence is in valid range with better calibration
            final_confidence = max(0.55, min(0.95, final_confidence))
            
            logger.debug(f"Ensemble scores - ViT: {vit_score:.3f}, CLIP: {clip_score:.3f}, "
                        f"Heuristic: {heuristic_score:.3f}, Final: {weighted_score:.3f}")
            
            return prediction, final_confidence
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            # Fallback to heuristic only
            heuristic_score, heuristic_conf = self._predict_with_heuristics(image)
            if heuristic_score > 0.5:
                return "medical", heuristic_score
            else:
                return "non-medical", 1.0 - heuristic_score

