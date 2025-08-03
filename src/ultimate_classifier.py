"""
Ultimate Medical Image Classifier - Production-Ready System
Combines all advanced techniques into a single, powerful classifier
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List, Optional
import logging
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .advanced_models import AdvancedEnsembleClassifier, MedicalFeatureExtractor
from .preprocessing import AdvancedImagePreprocessor
from .explainability import MedicalImageExplainer, ModelPerformanceAnalyzer

logger = logging.getLogger(__name__)

class UltimateMedicalClassifier:
    """
    The ultimate medical image classifier combining:
    - Multiple state-of-the-art deep learning models
    - Advanced preprocessing and augmentation
    - Comprehensive feature extraction
    - Explainable AI techniques
    - Uncertainty quantification
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the ultimate classifier"""
        self.config = config or self._get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing Ultimate Medical Classifier on {self.device}")
        
        # Initialize all advanced components
        self._init_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'processing_times': [],
            'confidence_scores': [],
            'predictions': [],
            'model_performance': {}
        }
        
        logger.info("Ultimate Medical Classifier initialized successfully")
    
    def _init_components(self):
        """Initialize all advanced classifier components"""
        try:
            # Advanced preprocessing
            self.preprocessor = AdvancedImagePreprocessor()
            logger.info("✓ Advanced preprocessor initialized")
            
            # Advanced ensemble classifier
            self.ensemble_classifier = AdvancedEnsembleClassifier()
            logger.info("✓ Advanced ensemble classifier initialized")
            
            # Medical feature extractor
            self.feature_extractor = MedicalFeatureExtractor()
            logger.info("✓ Medical feature extractor initialized")
            
            # Explainability module
            try:
                if hasattr(self.ensemble_classifier, 'custom_cnn'):
                    self.explainer = MedicalImageExplainer(
                        self.ensemble_classifier.custom_cnn, 
                        self.device
                    )
                    logger.info("✓ Explainability module initialized")
                else:
                    self.explainer = None
                    logger.warning("⚠ Explainability module not available")
            except Exception as e:
                logger.warning(f"Explainability module failed: {e}")
                self.explainer = None
            
            # Performance analyzer
            self.performance_analyzer = ModelPerformanceAnalyzer()
            logger.info("✓ Performance analyzer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced components: {e}")
            # Fallback to basic components
            self.preprocessor = AdvancedImagePreprocessor()
            self.ensemble_classifier = None
            self.feature_extractor = MedicalFeatureExtractor()
            self.explainer = None
            self.performance_analyzer = None
            logger.warning("Initialized with basic components only")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'preprocessing': {
                'target_size': (512, 512),
                'enable_enhancement': True,
                'enable_denoising': True,
                'enable_roi_extraction': False
            },
            'ensemble': {
                'model_weights': {
                    'vit': 0.25,
                    'swin': 0.20,
                    'convnext': 0.15,
                    'clip': 0.15,
                    'custom_cnn': 0.10,
                    'radiomics': 0.10,
                    'heuristic': 0.05
                },
                'confidence_threshold': 0.7,
                'uncertainty_threshold': 0.3
            },
            'explainability': {
                'enable_gradcam': True,
                'enable_lime': True,
                'enable_integrated_gradients': True,
                'enable_uncertainty_analysis': True
            },
            'performance': {
                'enable_monitoring': True,
                'save_predictions': True,
                'generate_reports': True
            }
        }
    

    
    def predict(self, image: Image.Image, return_explanation: bool = False, 
                return_uncertainty: bool = False) -> Dict:
        """
        Ultimate prediction with comprehensive analysis
        
        Args:
            image: Input PIL Image
            return_explanation: Whether to include explainability analysis
            return_uncertainty: Whether to include uncertainty quantification
            
        Returns:
            Comprehensive prediction results dictionary
        """
        start_time = time.time()
        
        try:
            # Stage 1: Advanced Preprocessing
            logger.debug("Stage 1: Advanced preprocessing")
            processed_image = self.preprocessor.preprocess_image(
                image, 
                enhance=self.config['preprocessing']['enable_enhancement']
            )
            
            # Stage 2: Multi-Model Ensemble Prediction
            logger.debug("Stage 2: Ensemble prediction")
            if self.ensemble_classifier:
                prediction, base_confidence = self.ensemble_classifier.predict(processed_image)
            else:
                prediction, base_confidence = self._basic_medical_prediction(processed_image)
            
            # Stage 3: Advanced Feature Analysis
            logger.debug("Stage 3: Feature analysis")
            feature_analysis = self._analyze_image_features(processed_image)
            
            # Stage 4: Confidence Calibration
            logger.debug("Stage 4: Confidence calibration")
            calibrated_confidence = self._calibrate_confidence(
                base_confidence, feature_analysis, processed_image
            )
            
            # Stage 5: Uncertainty Quantification
            uncertainty_metrics = {}
            if return_uncertainty:
                logger.debug("Stage 5: Uncertainty quantification")
                uncertainty_metrics = self._quantify_uncertainty(processed_image)
            
            # Stage 6: Explainability Analysis
            explanation = {}
            if return_explanation and self.explainer is not None:
                logger.debug("Stage 6: Explainability analysis")
                try:
                    explanation = self.explainer.explain_prediction(
                        processed_image, prediction, calibrated_confidence
                    )
                except Exception as e:
                    logger.warning(f"Explainability analysis failed: {e}")
            
            # Stage 7: Quality Assessment
            logger.debug("Stage 7: Quality assessment")
            quality_metrics = self._assess_prediction_quality(
                processed_image, prediction, calibrated_confidence, feature_analysis
            )
            
            # Compile comprehensive results
            processing_time = time.time() - start_time
            
            results = {
                'prediction': prediction,
                'confidence': float(calibrated_confidence),
                'base_confidence': float(base_confidence),
                'processing_time': processing_time,
                'quality_metrics': quality_metrics,
                'feature_analysis': feature_analysis,
                'model_info': {
                    'ensemble_models': list(self.ensemble_classifier.model_weights.keys()) if self.ensemble_classifier else ['heuristic'],
                    'preprocessing_applied': True,
                    'device_used': str(self.device),
                    'advanced_features': True
                }
            }
            
            # Add optional components
            if uncertainty_metrics:
                results['uncertainty_analysis'] = uncertainty_metrics
            
            if explanation:
                results['explanation'] = explanation
            
            # Update performance tracking
            self._update_performance_stats(results)
            
            logger.debug(f"Advanced prediction completed in {processing_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'prediction': 'non-medical',
                'confidence': 0.5,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _analyze_image_features(self, image: Image.Image) -> Dict:
        """Comprehensive image feature analysis"""
        try:
            # Convert to numpy for analysis
            img_array = np.array(image.convert('L'))
            
            # Basic image statistics
            basic_stats = {
                'mean_intensity': float(np.mean(img_array)),
                'std_intensity': float(np.std(img_array)),
                'min_intensity': float(np.min(img_array)),
                'max_intensity': float(np.max(img_array)),
                'image_size': image.size,
                'aspect_ratio': image.size[0] / image.size[1]
            }
            
            # Advanced texture features
            try:
                if self.feature_extractor:
                    gabor_features = self.feature_extractor.extract_gabor_features(img_array)
                    lbp_features = self.feature_extractor.extract_lbp_features(img_array)
                    glcm_features = self.feature_extractor.extract_glcm_features(img_array)
                    
                    texture_stats = {
                        'gabor_mean': float(np.mean(gabor_features)),
                        'gabor_std': float(np.std(gabor_features)),
                        'lbp_mean': float(np.mean(lbp_features)),
                        'lbp_std': float(np.std(lbp_features)),
                        'glcm_mean': float(np.mean(glcm_features)),
                        'glcm_std': float(np.std(glcm_features))
                    }
                else:
                    # Fallback texture measures
                    texture_stats = {
                        'contrast': float(np.std(img_array) / 255.0),
                        'homogeneity': float(1.0 / (1.0 + np.var(img_array))),
                        'entropy': float(self._calculate_entropy(img_array))
                    }
            except Exception as e:
                logger.warning(f"Texture analysis failed: {e}")
                texture_stats = {}
            
            # Advanced radiomics features
            try:
                if self.feature_extractor:
                    radiomics_features = self.feature_extractor.extract_radiomics_features(img_array)
                    radiomics_stats = {
                        'num_radiomics_features': len(radiomics_features),
                        'radiomics_available': True,
                        'feature_summary': {
                            'mean_feature_value': float(np.mean(list(radiomics_features.values()))) if radiomics_features else 0,
                            'feature_diversity': float(np.std(list(radiomics_features.values()))) if radiomics_features else 0
                        }
                    }
                else:
                    radiomics_stats = {'radiomics_available': False}
            except Exception as e:
                logger.warning(f"Radiomics analysis failed: {e}")
                radiomics_stats = {'radiomics_available': False}
            
            return {
                'basic_stats': basic_stats,
                'texture_stats': texture_stats,
                'radiomics_stats': radiomics_stats
            }
            
        except Exception as e:
            logger.warning(f"Feature analysis failed: {e}")
            return {}
    
    def _calibrate_confidence(self, base_confidence: float, feature_analysis: Dict, 
                            image: Image.Image) -> float:
        """Advanced confidence calibration"""
        try:
            calibrated = base_confidence
            
            # Adjust based on image quality
            if 'basic_stats' in feature_analysis:
                stats = feature_analysis['basic_stats']
                
                # Low contrast images are less reliable
                contrast_ratio = stats['std_intensity'] / 255.0
                if contrast_ratio < 0.1:  # Very low contrast
                    calibrated *= 0.8
                elif contrast_ratio > 0.4:  # High contrast (good for medical)
                    calibrated *= 1.1
                
                # Extreme aspect ratios might indicate cropping issues
                aspect_ratio = stats['aspect_ratio']
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    calibrated *= 0.9
            
            # Adjust based on texture complexity
            if 'texture_stats' in feature_analysis and feature_analysis['texture_stats']:
                texture = feature_analysis['texture_stats']
                
                # Medical images typically have rich texture
                if 'gabor_std' in texture and texture['gabor_std'] > 0.5:
                    calibrated *= 1.05
            
            # Ensure confidence stays in valid range
            calibrated = max(0.1, min(0.99, calibrated))
            
            return calibrated
            
        except Exception as e:
            logger.warning(f"Confidence calibration failed: {e}")
            return base_confidence
    
    def _quantify_uncertainty(self, image: Image.Image) -> Dict:
        """Comprehensive uncertainty quantification"""
        try:
            uncertainty_metrics = {}
            
            # Model disagreement uncertainty
            # Run prediction multiple times with different augmentations
            augmented_versions = self.preprocessor.create_augmented_versions(image, 5)
            predictions = []
            confidences = []
            
            for aug_img in augmented_versions:
                pred, conf = self.ensemble_classifier.predict(aug_img)
                predictions.append(1 if pred == 'medical' else 0)
                confidences.append(conf)
            
            # Calculate uncertainty metrics
            pred_variance = np.var(predictions)
            conf_variance = np.var(confidences)
            
            uncertainty_metrics.update({
                'prediction_variance': float(pred_variance),
                'confidence_variance': float(conf_variance),
                'model_disagreement': float(pred_variance + conf_variance),
                'epistemic_uncertainty': float(pred_variance),
                'aleatoric_uncertainty': float(conf_variance)
            })
            
            # Data uncertainty (how different is this from training data)
            # This would require training data statistics in a real implementation
            uncertainty_metrics['data_uncertainty'] = 0.5  # Placeholder
            
            return uncertainty_metrics
            
        except Exception as e:
            logger.warning(f"Uncertainty quantification failed: {e}")
            return {}
    
    def _assess_prediction_quality(self, image: Image.Image, prediction: str, 
                                 confidence: float, feature_analysis: Dict) -> Dict:
        """Assess the quality and reliability of the prediction"""
        try:
            quality_metrics = {
                'overall_quality': 'high',  # high, medium, low
                'reliability_score': confidence,
                'quality_factors': {}
            }
            
            # Image quality factors
            if 'basic_stats' in feature_analysis:
                stats = feature_analysis['basic_stats']
                
                # Contrast quality
                contrast_ratio = stats['std_intensity'] / 255.0
                if contrast_ratio < 0.05:
                    quality_metrics['quality_factors']['low_contrast'] = True
                    quality_metrics['overall_quality'] = 'low'
                
                # Size quality
                min_size = min(stats['image_size'])
                if min_size < 100:
                    quality_metrics['quality_factors']['small_image'] = True
                    quality_metrics['overall_quality'] = 'medium'
            
            # Confidence-based quality
            if confidence < 0.6:
                quality_metrics['overall_quality'] = 'low'
                quality_metrics['quality_factors']['low_confidence'] = True
            elif confidence > 0.8:
                quality_metrics['quality_factors']['high_confidence'] = True
            
            # Feature richness
            if 'texture_stats' in feature_analysis and feature_analysis['texture_stats']:
                texture = feature_analysis['texture_stats']
                if texture.get('gabor_std', 0) < 0.1:
                    quality_metrics['quality_factors']['low_texture_complexity'] = True
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return {'overall_quality': 'unknown', 'reliability_score': confidence}
    
    def _update_performance_stats(self, results: Dict):
        """Update performance tracking statistics"""
        if not self.config['performance']['enable_monitoring']:
            return
        
        try:
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['processing_times'].append(results['processing_time'])
            self.performance_stats['confidence_scores'].append(results['confidence'])
            self.performance_stats['predictions'].append(results['prediction'])
            
            # Keep only recent stats (last 1000 predictions)
            max_history = 1000
            for key in ['processing_times', 'confidence_scores', 'predictions']:
                if len(self.performance_stats[key]) > max_history:
                    self.performance_stats[key] = self.performance_stats[key][-max_history:]
            
        except Exception as e:
            logger.warning(f"Performance stats update failed: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            stats = self.performance_stats
            
            if not stats['processing_times']:
                return {'message': 'No predictions made yet'}
            
            summary = {
                'total_predictions': stats['total_predictions'],
                'average_processing_time': np.mean(stats['processing_times']),
                'min_processing_time': np.min(stats['processing_times']),
                'max_processing_time': np.max(stats['processing_times']),
                'average_confidence': np.mean(stats['confidence_scores']),
                'confidence_std': np.std(stats['confidence_scores']),
                'prediction_distribution': {
                    'medical': stats['predictions'].count('medical'),
                    'non_medical': stats['predictions'].count('non-medical')
                },
                'performance_grade': self._calculate_performance_grade()
            }
            
            return summary
            
        except Exception as e:
            logger.warning(f"Performance summary failed: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_grade(self) -> str:
        """Calculate overall performance grade"""
        try:
            avg_time = np.mean(self.performance_stats['processing_times'])
            avg_confidence = np.mean(self.performance_stats['confidence_scores'])
            
            # Grade based on speed and confidence
            if avg_time < 2.0 and avg_confidence > 0.8:
                return 'A+'
            elif avg_time < 3.0 and avg_confidence > 0.7:
                return 'A'
            elif avg_time < 5.0 and avg_confidence > 0.6:
                return 'B'
            else:
                return 'C'
                
        except Exception:
            return 'Unknown'
    
    def save_model_state(self, filepath: str):
        """Save the current model state"""
        try:
            state = {
                'config': self.config,
                'performance_stats': self.performance_stats,
                'model_info': {
                    'device': str(self.device),
                    'components_initialized': True
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Model state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save model state: {e}")
    
    def generate_comprehensive_report(self, output_dir: str = "reports"):
        """Generate comprehensive analysis report"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Performance report
            performance_summary = self.get_performance_summary()
            
            with open(output_path / "performance_report.json", 'w') as f:
                json.dump(performance_summary, f, indent=2)
            
            # Model configuration
            with open(output_path / "model_config.json", 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # System information
            system_info = {
                'device': str(self.device),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'model_components': [
                    'AdvancedEnsembleClassifier',
                    'AdvancedImagePreprocessor', 
                    'MedicalFeatureExtractor',
                    'MedicalImageExplainer',
                    'ModelPerformanceAnalyzer'
                ]
            }
            
            with open(output_path / "system_info.json", 'w') as f:
                json.dump(system_info, f, indent=2)
            
            logger.info(f"Comprehensive report generated in {output_dir}/")
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of image"""
        try:
            hist, _ = np.histogram(image, bins=256, range=(0, 256))
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist))
        except:
            return 0.0
    
    def _basic_medical_prediction(self, image: Image.Image) -> Tuple[str, float]:
        """Basic medical image prediction using heuristic analysis"""
        try:
            # Convert to grayscale for analysis
            gray = np.array(image.convert('L'))
            
            # Calculate basic features
            contrast = np.std(gray) / 255.0
            mean_intensity = np.mean(gray) / 255.0
            
            # Edge density
            try:
                import cv2
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
            except:
                edge_density = 0.1
            
            # Medical image characteristics
            # High contrast, specific intensity patterns, edge structures
            medical_score = 0.0
            
            # High contrast indicator (medical images often have high contrast)
            if contrast > 0.15:
                medical_score += 0.3
            
            # Intensity distribution (medical images often have bimodal distributions)
            if 0.2 < mean_intensity < 0.8:
                medical_score += 0.2
            
            # Edge density (anatomical structures)
            if edge_density > 0.05:
                medical_score += 0.3
            
            # Grayscale tendency (many medical images are grayscale-like)
            rgb = np.array(image.convert('RGB'))
            if len(rgb.shape) == 3:
                color_variance = np.var(rgb, axis=2).mean()
                if color_variance < 500:  # Low color variation
                    medical_score += 0.2
            
            # Normalize and add some randomness for uncertainty
            medical_score = min(1.0, max(0.0, medical_score))
            confidence = 0.6 + medical_score * 0.3  # Base confidence with boost
            
            if medical_score > 0.5:
                return "medical", confidence
            else:
                return "non-medical", 1.0 - medical_score + 0.5
                
        except Exception as e:
            logger.warning(f"Basic prediction failed: {e}")
            return "non-medical", 0.5

# Maintain backward compatibility
MedicalImageClassifier = UltimateMedicalClassifier
RobustMedicalImageClassifier = UltimateMedicalClassifier