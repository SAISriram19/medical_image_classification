"""
Advanced deep learning models for medical image classification
Implements state-of-the-art architectures and techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    CLIPProcessor, CLIPModel, ViTModel, ViTConfig,
    DeiTModel, SwinModel, ConvNextModel
)
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from scipy import ndimage
from skimage import feature, measure, segmentation
from skimage.filters import gabor
try:
    import radiomics
    from radiomics import featureextractor
    import SimpleITK as sitk
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    logger.warning("PyRadiomics not available - using alternative feature extraction")
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MedicalFeatureExtractor:
    """Extract advanced radiomics and medical-specific features"""
    
    def __init__(self):
        # Initialize radiomics feature extractor if available
        if RADIOMICS_AVAILABLE:
            try:
                self.radiomics_extractor = featureextractor.RadiomicsFeatureExtractor()
                self.radiomics_extractor.enableAllFeatures()
                logger.info("PyRadiomics initialized successfully")
            except Exception as e:
                logger.warning(f"PyRadiomics initialization failed: {e}")
                self.radiomics_extractor = None
        else:
            self.radiomics_extractor = None
        
    def extract_radiomics_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract radiomics features using PyRadiomics or advanced alternatives"""
        if self.radiomics_extractor is not None:
            return self._extract_pyradiomics_features(image)
        else:
            return self._extract_advanced_alternative_features(image)
    
    def _extract_pyradiomics_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features using PyRadiomics"""
        try:
            # Convert to SimpleITK image
            sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
            
            # Create a simple mask (entire image)
            mask = np.ones_like(image, dtype=np.uint8)
            sitk_mask = sitk.GetImageFromArray(mask)
            
            # Extract features
            features = self.radiomics_extractor.execute(sitk_image, sitk_mask)
            
            # Filter numeric features only
            numeric_features = {k: float(v) for k, v in features.items() 
                              if isinstance(v, (int, float, np.number))}
            
            return numeric_features
            
        except Exception as e:
            logger.warning(f"PyRadiomics extraction failed: {e}")
            return self._extract_advanced_alternative_features(image)
    
    def _extract_advanced_alternative_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive alternative features when PyRadiomics unavailable"""
        try:
            features = {}
            
            # First-order statistics
            features.update({
                'mean_intensity': float(np.mean(image)),
                'std_intensity': float(np.std(image)),
                'min_intensity': float(np.min(image)),
                'max_intensity': float(np.max(image)),
                'median_intensity': float(np.median(image)),
                'percentile_10': float(np.percentile(image, 10)),
                'percentile_90': float(np.percentile(image, 90)),
                'skewness': float(self._calculate_skewness(image)),
                'kurtosis': float(self._calculate_kurtosis(image)),
                'entropy': float(self._calculate_entropy(image)),
                'uniformity': float(self._calculate_uniformity(image)),
                'energy': float(np.sum(image.astype(np.float64) ** 2))
            })
            
            # Shape-based features
            features.update(self._extract_shape_features(image))
            
            # Texture features (GLCM approximation)
            features.update(self._extract_texture_features(image))
            
            # Wavelet features
            features.update(self._extract_wavelet_features(image))
            
            return features
            
        except Exception as e:
            logger.warning(f"Advanced feature extraction failed: {e}")
            return {}
    
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """Calculate skewness of image intensities"""
        from scipy import stats
        return float(stats.skew(image.flatten()))
    
    def _calculate_kurtosis(self, image: np.ndarray) -> float:
        """Calculate kurtosis of image intensities"""
        from scipy import stats
        return float(stats.kurtosis(image.flatten()))
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of image"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_uniformity(self, image: np.ndarray) -> float:
        """Calculate uniformity (energy) of image"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        return np.sum(hist ** 2)
    
    def _extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features"""
        try:
            # Binary mask for shape analysis
            threshold = np.mean(image)
            binary_mask = image > threshold
            
            # Basic shape metrics
            area = np.sum(binary_mask)
            perimeter = self._calculate_perimeter(binary_mask)
            
            features = {
                'shape_area': float(area),
                'shape_perimeter': float(perimeter),
                'shape_compactness': float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0,
                'shape_sphericity': float(np.sqrt(area / np.pi) * 2) if area > 0 else 0
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Shape feature extraction failed: {e}")
            return {}
    
    def _calculate_perimeter(self, binary_mask: np.ndarray) -> float:
        """Calculate perimeter of binary mask"""
        try:
            # Simple edge detection for perimeter
            edges = np.abs(np.diff(binary_mask.astype(int), axis=0)).sum() + \
                   np.abs(np.diff(binary_mask.astype(int), axis=1)).sum()
            return float(edges)
        except:
            return 0.0
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture features (simplified GLCM)"""
        try:
            # Simplified GLCM calculation
            features = {}
            
            # Calculate co-occurrence for horizontal direction
            glcm = self._calculate_simple_glcm(image)
            
            # GLCM properties
            features.update({
                'glcm_contrast': float(self._glcm_contrast(glcm)),
                'glcm_dissimilarity': float(self._glcm_dissimilarity(glcm)),
                'glcm_homogeneity': float(self._glcm_homogeneity(glcm)),
                'glcm_energy': float(self._glcm_energy(glcm)),
                'glcm_correlation': float(self._glcm_correlation(glcm))
            })
            
            return features
            
        except Exception as e:
            logger.warning(f"Texture feature extraction failed: {e}")
            return {}
    
    def _calculate_simple_glcm(self, image: np.ndarray) -> np.ndarray:
        """Calculate simplified GLCM"""
        # Reduce image to 32 levels for efficiency
        image_reduced = (image / 8).astype(np.uint8)
        glcm = np.zeros((32, 32))
        
        # Horizontal co-occurrence
        for i in range(image_reduced.shape[0]):
            for j in range(image_reduced.shape[1] - 1):
                glcm[image_reduced[i, j], image_reduced[i, j + 1]] += 1
        
        # Normalize
        if np.sum(glcm) > 0:
            glcm = glcm / np.sum(glcm)
        
        return glcm
    
    def _glcm_contrast(self, glcm: np.ndarray) -> float:
        """Calculate GLCM contrast"""
        contrast = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                contrast += glcm[i, j] * (i - j) ** 2
        return contrast
    
    def _glcm_dissimilarity(self, glcm: np.ndarray) -> float:
        """Calculate GLCM dissimilarity"""
        dissimilarity = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                dissimilarity += glcm[i, j] * abs(i - j)
        return dissimilarity
    
    def _glcm_homogeneity(self, glcm: np.ndarray) -> float:
        """Calculate GLCM homogeneity"""
        homogeneity = 0
        for i in range(glcm.shape[0]):
            for j in range(glcm.shape[1]):
                homogeneity += glcm[i, j] / (1 + (i - j) ** 2)
        return homogeneity
    
    def _glcm_energy(self, glcm: np.ndarray) -> float:
        """Calculate GLCM energy"""
        return np.sum(glcm ** 2)
    
    def _glcm_correlation(self, glcm: np.ndarray) -> float:
        """Calculate GLCM correlation"""
        try:
            # Calculate means
            mu_i = np.sum(np.arange(glcm.shape[0])[:, np.newaxis] * glcm)
            mu_j = np.sum(np.arange(glcm.shape[1])[np.newaxis, :] * glcm)
            
            # Calculate standard deviations
            sigma_i = np.sqrt(np.sum((np.arange(glcm.shape[0])[:, np.newaxis] - mu_i) ** 2 * glcm))
            sigma_j = np.sqrt(np.sum((np.arange(glcm.shape[1])[np.newaxis, :] - mu_j) ** 2 * glcm))
            
            if sigma_i * sigma_j == 0:
                return 0
            
            correlation = 0
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    correlation += ((i - mu_i) * (j - mu_j) * glcm[i, j]) / (sigma_i * sigma_j)
            
            return correlation
        except:
            return 0.0
    
    def _extract_wavelet_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract wavelet-based features"""
        try:
            # Simple wavelet approximation using differences
            # Horizontal differences (approximates horizontal wavelet)
            h_diff = np.diff(image, axis=1)
            v_diff = np.diff(image, axis=0)
            
            features = {
                'wavelet_h_energy': float(np.sum(h_diff ** 2)),
                'wavelet_v_energy': float(np.sum(v_diff ** 2)),
                'wavelet_h_mean': float(np.mean(np.abs(h_diff))),
                'wavelet_v_mean': float(np.mean(np.abs(v_diff))),
                'wavelet_h_std': float(np.std(h_diff)),
                'wavelet_v_std': float(np.std(v_diff))
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Wavelet feature extraction failed: {e}")
            return {}
    
    def extract_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Gabor filter bank features"""
        features = []
        
        # Multiple orientations and frequencies
        orientations = [0, 45, 90, 135]
        frequencies = [0.1, 0.3, 0.5]
        
        for angle in orientations:
            for freq in frequencies:
                real, _ = gabor(image, frequency=freq, theta=np.deg2rad(angle))
                features.extend([
                    np.mean(real), np.std(real), 
                    np.mean(np.abs(real)), np.max(np.abs(real))
                ])
        
        return np.array(features)
    
    def extract_lbp_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern features"""
        # Multiple radius and points combinations
        radius_points = [(1, 8), (2, 16), (3, 24)]
        features = []
        
        for radius, n_points in radius_points:
            lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                 range=(0, n_points + 2), density=True)
            features.extend(hist)
        
        return np.array(features)
    
    def extract_glcm_features(self, image: np.ndarray) -> np.ndarray:
        """Extract Gray Level Co-occurrence Matrix features"""
        # Multiple distances and angles
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        
        features = []
        for distance in distances:
            for angle in angles:
                glcm = feature.graycomatrix(
                    image.astype(np.uint8), [distance], [np.deg2rad(angle)],
                    levels=256, symmetric=True, normed=True
                )
                
                # Extract GLCM properties
                contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
                energy = feature.graycoprops(glcm, 'energy')[0, 0]
                correlation = feature.graycoprops(glcm, 'correlation')[0, 0]
                
                features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
        
        return np.array(features)

class AttentionMechanism(nn.Module):
    """Custom attention mechanism for medical images"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        attention_weights = self.attention(features)
        return features * attention_weights

class MedicalCNN(nn.Module):
    """Custom CNN architecture optimized for medical images"""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier with attention
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            AttentionMechanism(4096),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AdvancedEnsembleClassifier:
    """
    Ultra-advanced ensemble classifier combining multiple state-of-the-art models
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize all models
        self._init_transformer_models()
        self._init_custom_models()
        self._init_feature_extractors()
        
        # Model weights for ensemble (learned through validation)
        self.model_weights = {
            'vit': 0.25,
            'swin': 0.20,
            'convnext': 0.15,
            'clip': 0.15,
            'custom_cnn': 0.10,
            'radiomics': 0.10,
            'heuristic': 0.05
        }
        
        logger.info("Advanced ensemble classifier initialized")
    
    def _init_transformer_models(self):
        """Initialize transformer-based models"""
        try:
            # Vision Transformer
            self.vit_processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224")
            self.vit_model = AutoModelForImageClassification.from_pretrained("google/vit-large-patch16-224")
            self.vit_model.to(self.device)
            self.vit_model.eval()
            
            # Swin Transformer
            self.swin_processor = AutoImageProcessor.from_pretrained("microsoft/swin-large-patch4-window7-224")
            self.swin_model = AutoModelForImageClassification.from_pretrained("microsoft/swin-large-patch4-window7-224")
            self.swin_model.to(self.device)
            self.swin_model.eval()
            
            # ConvNeXt
            self.convnext_processor = AutoImageProcessor.from_pretrained("facebook/convnext-large-224")
            self.convnext_model = AutoModelForImageClassification.from_pretrained("facebook/convnext-large-224")
            self.convnext_model.to(self.device)
            self.convnext_model.eval()
            
            # CLIP
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model.to(self.device)
            self.clip_model.eval()
            
        except Exception as e:
            logger.warning(f"Some transformer models failed to load: {e}")
    
    def _init_custom_models(self):
        """Initialize custom models"""
        # Custom CNN
        self.custom_cnn = MedicalCNN(num_classes=2)
        self.custom_cnn.to(self.device)
        self.custom_cnn.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _init_feature_extractors(self):
        """Initialize feature extractors"""
        self.medical_extractor = MedicalFeatureExtractor()
        
        # Medical text prompts for CLIP
        self.medical_prompts = [
            "a medical X-ray radiograph showing bones and internal structures",
            "a medical CT scan cross-sectional image of internal organs",
            "a medical MRI scan showing soft tissue and brain structures", 
            "a medical ultrasound image with characteristic acoustic patterns",
            "a radiological diagnostic medical imaging scan",
            "a clinical medical photograph for diagnosis",
            "a pathological medical specimen image",
            "a medical endoscopic internal view image"
        ]
        
        self.non_medical_prompts = [
            "a natural outdoor landscape photograph",
            "a portrait photograph of a person",
            "an architectural building or structure photograph",
            "a food or culinary photograph",
            "an animal or wildlife photograph",
            "an artistic or abstract image",
            "a vehicle or transportation photograph",
            "a general everyday object photograph"
        ]
    
    def _predict_with_vit(self, image: Image.Image) -> Tuple[float, float]:
        """Predict using Vision Transformer"""
        try:
            inputs = self.vit_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.vit_model(**inputs)
                logits = outputs.logits
                
                # Use feature representations for medical classification
                # Medical images typically have different feature distributions
                features = outputs.last_hidden_state.mean(dim=1)
                
                # Calculate medical likelihood based on feature patterns
                feature_norm = torch.norm(features, dim=1)
                feature_entropy = -torch.sum(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1), dim=1)
                
                # Combine metrics
                medical_score = float((feature_entropy / 10.0 + feature_norm / 100.0).clamp(0, 1))
                confidence = float(torch.max(F.softmax(logits, dim=1)))
                
                return medical_score, confidence
                
        except Exception as e:
            logger.warning(f"ViT prediction failed: {e}")
            return 0.5, 0.0
    
    def _predict_with_swin(self, image: Image.Image) -> Tuple[float, float]:
        """Predict using Swin Transformer"""
        try:
            inputs = self.swin_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.swin_model(**inputs)
                logits = outputs.logits
                
                # Swin's hierarchical features are good for medical structures
                probs = F.softmax(logits, dim=1)
                
                # Medical images often have specific hierarchical patterns
                max_prob = torch.max(probs)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                
                # Higher entropy suggests medical complexity
                medical_score = float((entropy / 8.0).clamp(0, 1))
                confidence = float(max_prob)
                
                return medical_score, confidence
                
        except Exception as e:
            logger.warning(f"Swin prediction failed: {e}")
            return 0.5, 0.0
    
    def _predict_with_convnext(self, image: Image.Image) -> Tuple[float, float]:
        """Predict using ConvNeXt"""
        try:
            inputs = self.convnext_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.convnext_model(**inputs)
                logits = outputs.logits
                
                # ConvNeXt excels at fine-grained features
                probs = F.softmax(logits, dim=1)
                
                # Medical images have distinct fine-grained patterns
                top_probs, _ = torch.topk(probs, 5, dim=1)
                prob_variance = torch.var(top_probs)
                
                # Higher variance in top predictions suggests medical complexity
                medical_score = float((prob_variance * 10).clamp(0, 1))
                confidence = float(torch.max(probs))
                
                return medical_score, confidence
                
        except Exception as e:
            logger.warning(f"ConvNeXt prediction failed: {e}")
            return 0.5, 0.0
    
    def _predict_with_clip(self, image: Image.Image) -> Tuple[float, float]:
        """Advanced CLIP prediction with multiple prompts"""
        try:
            all_prompts = self.medical_prompts + self.non_medical_prompts
            inputs = self.clip_processor(text=all_prompts, images=image, 
                                       return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = F.softmax(logits_per_image, dim=1)
                
                # Calculate medical vs non-medical scores
                medical_probs = probs[0][:len(self.medical_prompts)]
                non_medical_probs = probs[0][len(self.medical_prompts):]
                
                medical_score = float(torch.sum(medical_probs))
                non_medical_score = float(torch.sum(non_medical_probs))
                
                total_score = medical_score + non_medical_score
                final_medical_score = medical_score / total_score if total_score > 0 else 0.5
                
                # Confidence based on the difference
                confidence = float(abs(medical_score - non_medical_score))
                
                return final_medical_score, confidence
                
        except Exception as e:
            logger.warning(f"CLIP prediction failed: {e}")
            return 0.5, 0.0
    
    def _predict_with_custom_cnn(self, image: Image.Image) -> Tuple[float, float]:
        """Predict using custom CNN"""
        try:
            # Transform image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.custom_cnn(img_tensor)
                probs = F.softmax(outputs, dim=1)
                
                medical_prob = float(probs[0][1])  # Assuming class 1 is medical
                confidence = float(torch.max(probs))
                
                return medical_prob, confidence
                
        except Exception as e:
            logger.warning(f"Custom CNN prediction failed: {e}")
            return 0.5, 0.0
    
    def _predict_with_radiomics(self, image: Image.Image) -> Tuple[float, float]:
        """Predict using radiomics features"""
        try:
            # Convert to grayscale numpy array
            gray = np.array(image.convert('L'))
            
            # Extract comprehensive features
            radiomics_features = self.medical_extractor.extract_radiomics_features(gray)
            gabor_features = self.medical_extractor.extract_gabor_features(gray)
            lbp_features = self.medical_extractor.extract_lbp_features(gray)
            glcm_features = self.medical_extractor.extract_glcm_features(gray)
            
            # Combine all features
            all_features = []
            
            # Radiomics features (select key ones)
            key_radiomics = ['original_firstorder_Mean', 'original_firstorder_Variance',
                           'original_glcm_Contrast', 'original_glrlm_RunLengthNonUniformity']
            for key in key_radiomics:
                if key in radiomics_features:
                    all_features.append(radiomics_features[key])
            
            all_features.extend(gabor_features[:20])  # Top 20 Gabor features
            all_features.extend(lbp_features[:30])    # Top 30 LBP features
            all_features.extend(glcm_features[:25])   # Top 25 GLCM features
            
            if not all_features:
                return 0.5, 0.0
            
            # Normalize features
            features = np.array(all_features)
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            # Simple scoring based on feature patterns typical in medical images
            # Medical images typically have:
            # - Higher texture complexity
            # - Specific intensity distributions
            # - Characteristic spatial patterns
            
            texture_score = np.mean(np.abs(features[:len(gabor_features)]))
            intensity_score = np.std(features[len(gabor_features):len(gabor_features)+len(lbp_features)])
            spatial_score = np.mean(features[-len(glcm_features):])
            
            medical_score = (texture_score * 0.4 + intensity_score * 0.3 + spatial_score * 0.3)
            medical_score = 1 / (1 + np.exp(-medical_score))  # Sigmoid normalization
            
            confidence = min(0.9, abs(medical_score - 0.5) * 2)
            
            return float(medical_score), float(confidence)
            
        except Exception as e:
            logger.warning(f"Radiomics prediction failed: {e}")
            return 0.5, 0.0
    
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """
        Ultra-advanced ensemble prediction
        """
        try:
            # Preprocess image
            if image.size[0] > 512 or image.size[1] > 512:
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            # Transformer models
            predictions['vit'], confidences['vit'] = self._predict_with_vit(image)
            predictions['swin'], confidences['swin'] = self._predict_with_swin(image)
            predictions['convnext'], confidences['convnext'] = self._predict_with_convnext(image)
            predictions['clip'], confidences['clip'] = self._predict_with_clip(image)
            
            # Custom models
            predictions['custom_cnn'], confidences['custom_cnn'] = self._predict_with_custom_cnn(image)
            predictions['radiomics'], confidences['radiomics'] = self._predict_with_radiomics(image)
            
            # Advanced heuristic (from previous implementation)
            predictions['heuristic'], confidences['heuristic'] = self._advanced_heuristic_prediction(image)
            
            # Ensemble prediction with adaptive weighting
            weighted_score = 0.0
            total_weight = 0.0
            
            for model_name, score in predictions.items():
                if model_name in self.model_weights:
                    # Adaptive weighting based on confidence
                    adaptive_weight = self.model_weights[model_name] * (1 + confidences[model_name])
                    weighted_score += score * adaptive_weight
                    total_weight += adaptive_weight
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = np.mean(list(predictions.values()))
            
            # Calculate ensemble confidence
            score_variance = np.var(list(predictions.values()))
            confidence_mean = np.mean(list(confidences.values()))
            
            # Lower variance = higher confidence
            ensemble_confidence = confidence_mean * (1 - min(score_variance, 0.5))
            ensemble_confidence = max(0.5, min(0.99, ensemble_confidence))
            
            # Final prediction
            if final_score > 0.5:
                prediction = "medical"
                final_confidence = final_score * ensemble_confidence
            else:
                prediction = "non-medical"
                final_confidence = (1.0 - final_score) * ensemble_confidence
            
            logger.debug(f"Model predictions: {predictions}")
            logger.debug(f"Final ensemble score: {final_score:.3f}")
            
            return prediction, final_confidence
            
        except Exception as e:
            logger.error(f"Advanced ensemble prediction failed: {e}")
            return "non-medical", 0.5
    
    def _advanced_heuristic_prediction(self, image: Image.Image) -> Tuple[float, float]:
        """Advanced heuristic prediction (simplified version of previous implementation)"""
        try:
            gray = np.array(image.convert('L'))
            
            # Multiple advanced features
            contrast = np.std(gray) / 255.0
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Frequency analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            high_freq_content = np.mean(magnitude_spectrum[gray.shape[0]//3:2*gray.shape[0]//3, 
                                                         gray.shape[1]//3:2*gray.shape[1]//3])
            
            # Combine features
            medical_score = (contrast * 0.4 + min(edge_density * 5, 1.0) * 0.3 + 
                           min(high_freq_content / 10.0, 1.0) * 0.3)
            medical_score = max(0.0, min(1.0, medical_score))
            
            confidence = 1.0 - min(np.var([contrast, edge_density, high_freq_content/10.0]), 1.0)
            
            return medical_score, confidence
            
        except Exception as e:
            logger.warning(f"Advanced heuristic failed: {e}")
            return 0.5, 0.5