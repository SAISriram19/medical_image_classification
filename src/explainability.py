"""
Advanced explainability and interpretability module for medical image classification
Implements GradCAM, LIME, SHAP, and custom visualization techniques
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import json
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import shap
from captum.attr import GradientShap, IntegratedGradients, Occlusion, LayerGradCam
from captum.attr import visualization as viz

logger = logging.getLogger(__name__)

class MedicalImageExplainer:
    """
    Comprehensive explainability suite for medical image classification
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Initialize explainability methods
        self._init_captum_methods()
        self._init_lime()
        
    def _init_captum_methods(self):
        """Initialize Captum attribution methods"""
        try:
            self.gradient_shap = GradientShap(self.model)
            self.integrated_gradients = IntegratedGradients(self.model)
            self.occlusion = Occlusion(self.model)
            
            # For GradCAM, we need to identify the target layer
            # This assumes the model has a 'features' attribute
            if hasattr(self.model, 'features'):
                target_layer = None
                for name, module in self.model.features.named_modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                        target_layer = module
                
                if target_layer is not None:
                    self.gradcam = LayerGradCam(self.model, target_layer)
                else:
                    self.gradcam = None
            else:
                self.gradcam = None
                
        except Exception as e:
            logger.warning(f"Failed to initialize some Captum methods: {e}")
    
    def _init_lime(self):
        """Initialize LIME explainer"""
        try:
            self.lime_explainer = lime_image.LimeImageExplainer()
        except Exception as e:
            logger.warning(f"Failed to initialize LIME: {e}")
            self.lime_explainer = None
    
    def explain_prediction(self, image: Image.Image, prediction: str, confidence: float) -> Dict:
        """
        Generate comprehensive explanation for a prediction
        """
        explanations = {
            'prediction': prediction,
            'confidence': confidence,
            'visual_explanations': {},
            'feature_importance': {},
            'uncertainty_analysis': {}
        }
        
        # Convert image for processing
        img_tensor = self._preprocess_image(image)
        
        # Generate different types of explanations
        explanations['visual_explanations']['gradcam'] = self._generate_gradcam(img_tensor, image)
        explanations['visual_explanations']['integrated_gradients'] = self._generate_integrated_gradients(img_tensor, image)
        explanations['visual_explanations']['occlusion'] = self._generate_occlusion_map(img_tensor, image)
        explanations['visual_explanations']['lime'] = self._generate_lime_explanation(image)
        
        # Feature importance analysis
        explanations['feature_importance'] = self._analyze_feature_importance(img_tensor)
        
        # Uncertainty analysis
        explanations['uncertainty_analysis'] = self._analyze_uncertainty(img_tensor)
        
        # Generate summary visualization
        explanations['summary_visualization'] = self._create_summary_visualization(
            image, explanations, prediction, confidence
        )
        
        return explanations
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        return img_tensor
    
    def _generate_gradcam(self, img_tensor: torch.Tensor, original_image: Image.Image) -> Optional[np.ndarray]:
        """Generate GradCAM visualization"""
        if self.gradcam is None:
            return None
        
        try:
            # Get the predicted class
            with torch.no_grad():
                output = self.model(img_tensor)
                predicted_class = output.argmax(dim=1).item()
            
            # Generate GradCAM
            attribution = self.gradcam.attribute(img_tensor, target=predicted_class)
            
            # Convert to numpy and resize to original image size
            gradcam_map = attribution.squeeze().cpu().numpy()
            gradcam_map = cv2.resize(gradcam_map, original_image.size)
            
            # Normalize to 0-1
            gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min())
            
            return gradcam_map
            
        except Exception as e:
            logger.warning(f"GradCAM generation failed: {e}")
            return None
    
    def _generate_integrated_gradients(self, img_tensor: torch.Tensor, original_image: Image.Image) -> Optional[np.ndarray]:
        """Generate Integrated Gradients visualization"""
        try:
            # Create baseline (black image)
            baseline = torch.zeros_like(img_tensor)
            
            # Get the predicted class
            with torch.no_grad():
                output = self.model(img_tensor)
                predicted_class = output.argmax(dim=1).item()
            
            # Generate Integrated Gradients
            attribution = self.integrated_gradients.attribute(
                img_tensor, baseline, target=predicted_class, n_steps=50
            )
            
            # Convert to numpy and process
            ig_map = attribution.squeeze().cpu().numpy()
            
            # Take the magnitude across channels
            ig_map = np.sqrt(np.sum(ig_map ** 2, axis=0))
            
            # Resize to original image size
            ig_map = cv2.resize(ig_map, original_image.size)
            
            # Normalize
            ig_map = (ig_map - ig_map.min()) / (ig_map.max() - ig_map.min())
            
            return ig_map
            
        except Exception as e:
            logger.warning(f"Integrated Gradients generation failed: {e}")
            return None
    
    def _generate_occlusion_map(self, img_tensor: torch.Tensor, original_image: Image.Image) -> Optional[np.ndarray]:
        """Generate occlusion sensitivity map"""
        try:
            # Get the predicted class
            with torch.no_grad():
                output = self.model(img_tensor)
                predicted_class = output.argmax(dim=1).item()
            
            # Generate occlusion map
            attribution = self.occlusion.attribute(
                img_tensor,
                strides=(3, 8, 8),
                target=predicted_class,
                sliding_window_shapes=(3, 15, 15),
                baselines=0
            )
            
            # Convert to numpy and process
            occlusion_map = attribution.squeeze().cpu().numpy()
            
            # Take the magnitude across channels
            occlusion_map = np.sqrt(np.sum(occlusion_map ** 2, axis=0))
            
            # Resize to original image size
            occlusion_map = cv2.resize(occlusion_map, original_image.size)
            
            # Normalize
            occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min())
            
            return occlusion_map
            
        except Exception as e:
            logger.warning(f"Occlusion map generation failed: {e}")
            return None
    
    def _generate_lime_explanation(self, image: Image.Image) -> Optional[Dict]:
        """Generate LIME explanation"""
        if self.lime_explainer is None:
            return None
        
        try:
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Define prediction function for LIME
            def predict_fn(images):
                predictions = []
                for img in images:
                    # Convert to PIL and preprocess
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    img_tensor = self._preprocess_image(pil_img)
                    
                    with torch.no_grad():
                        output = self.model(img_tensor)
                        probs = F.softmax(output, dim=1)
                        predictions.append(probs.cpu().numpy()[0])
                
                return np.array(predictions)
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                img_array,
                predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=1000
            )
            
            # Extract explanation data
            lime_data = {
                'segments': explanation.segments,
                'local_exp': dict(explanation.local_exp[1]),  # For class 1 (medical)
                'score': explanation.score
            }
            
            return lime_data
            
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            return None
    
    def _analyze_feature_importance(self, img_tensor: torch.Tensor) -> Dict:
        """Analyze feature importance across different scales"""
        try:
            feature_importance = {}
            
            # Get model features at different layers
            with torch.no_grad():
                # This is a simplified version - would need to be adapted based on model architecture
                if hasattr(self.model, 'features'):
                    x = img_tensor
                    layer_outputs = []
                    
                    for i, layer in enumerate(self.model.features):
                        x = layer(x)
                        if isinstance(layer, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
                            layer_outputs.append(x.clone())
                    
                    # Analyze activation patterns
                    for i, output in enumerate(layer_outputs):
                        activation_stats = {
                            'mean_activation': float(output.mean()),
                            'max_activation': float(output.max()),
                            'activation_sparsity': float((output == 0).float().mean()),
                            'channel_diversity': float(output.std(dim=1).mean())
                        }
                        feature_importance[f'layer_{i}'] = activation_stats
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
            return {}
    
    def _analyze_uncertainty(self, img_tensor: torch.Tensor) -> Dict:
        """Analyze prediction uncertainty using multiple techniques"""
        try:
            uncertainty_metrics = {}
            
            # Monte Carlo Dropout (if model has dropout layers)
            mc_predictions = []
            self.model.train()  # Enable dropout
            
            for _ in range(20):  # 20 forward passes
                with torch.no_grad():
                    output = self.model(img_tensor)
                    probs = F.softmax(output, dim=1)
                    mc_predictions.append(probs.cpu().numpy()[0])
            
            self.model.eval()  # Back to eval mode
            
            mc_predictions = np.array(mc_predictions)
            
            # Calculate uncertainty metrics
            uncertainty_metrics['predictive_entropy'] = float(-np.sum(
                np.mean(mc_predictions, axis=0) * np.log(np.mean(mc_predictions, axis=0) + 1e-8)
            ))
            
            uncertainty_metrics['aleatoric_uncertainty'] = float(np.mean(
                -np.sum(mc_predictions * np.log(mc_predictions + 1e-8), axis=1)
            ))
            
            uncertainty_metrics['epistemic_uncertainty'] = float(
                uncertainty_metrics['predictive_entropy'] - uncertainty_metrics['aleatoric_uncertainty']
            )
            
            uncertainty_metrics['prediction_variance'] = float(np.var(mc_predictions, axis=0).max())
            
            return uncertainty_metrics
            
        except Exception as e:
            logger.warning(f"Uncertainty analysis failed: {e}")
            return {}
    
    def _create_summary_visualization(self, image: Image.Image, explanations: Dict, 
                                    prediction: str, confidence: float) -> str:
        """Create comprehensive summary visualization"""
        try:
            fig = plt.figure(figsize=(20, 12))
            
            # Original image
            ax1 = plt.subplot(2, 4, 1)
            plt.imshow(image)
            plt.title(f'Original Image\nPrediction: {prediction}\nConfidence: {confidence:.3f}')
            plt.axis('off')
            
            # GradCAM
            if explanations['visual_explanations']['gradcam'] is not None:
                ax2 = plt.subplot(2, 4, 2)
                gradcam = explanations['visual_explanations']['gradcam']
                plt.imshow(image, alpha=0.7)
                plt.imshow(gradcam, cmap='jet', alpha=0.5)
                plt.title('GradCAM')
                plt.axis('off')
            
            # Integrated Gradients
            if explanations['visual_explanations']['integrated_gradients'] is not None:
                ax3 = plt.subplot(2, 4, 3)
                ig_map = explanations['visual_explanations']['integrated_gradients']
                plt.imshow(ig_map, cmap='hot')
                plt.title('Integrated Gradients')
                plt.axis('off')
            
            # Occlusion Map
            if explanations['visual_explanations']['occlusion'] is not None:
                ax4 = plt.subplot(2, 4, 4)
                occlusion_map = explanations['visual_explanations']['occlusion']
                plt.imshow(occlusion_map, cmap='viridis')
                plt.title('Occlusion Sensitivity')
                plt.axis('off')
            
            # Feature importance plot
            ax5 = plt.subplot(2, 4, 5)
            if explanations['feature_importance']:
                layers = list(explanations['feature_importance'].keys())
                mean_activations = [explanations['feature_importance'][layer]['mean_activation'] 
                                 for layer in layers]
                plt.bar(layers, mean_activations)
                plt.title('Layer Activations')
                plt.xticks(rotation=45)
            
            # Uncertainty analysis
            ax6 = plt.subplot(2, 4, 6)
            if explanations['uncertainty_analysis']:
                uncertainty_types = list(explanations['uncertainty_analysis'].keys())
                uncertainty_values = list(explanations['uncertainty_analysis'].values())
                plt.bar(uncertainty_types, uncertainty_values)
                plt.title('Uncertainty Analysis')
                plt.xticks(rotation=45)
            
            # LIME visualization (if available)
            if explanations['visual_explanations']['lime'] is not None:
                ax7 = plt.subplot(2, 4, 7)
                # Simplified LIME visualization
                plt.imshow(image)
                plt.title('LIME Explanation')
                plt.axis('off')
            
            # Summary statistics
            ax8 = plt.subplot(2, 4, 8)
            plt.axis('off')
            
            # Create summary text
            summary_text = f"""
            Prediction Summary:
            
            Class: {prediction}
            Confidence: {confidence:.3f}
            
            Uncertainty Metrics:
            """
            
            if explanations['uncertainty_analysis']:
                for key, value in explanations['uncertainty_analysis'].items():
                    summary_text += f"\n{key}: {value:.4f}"
            
            plt.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save visualization
            output_path = f'explanation_{prediction}_{confidence:.3f}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.warning(f"Summary visualization failed: {e}")
            return ""

class ModelPerformanceAnalyzer:
    """Comprehensive model performance analysis"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_model_performance(self, predictions: List, true_labels: List, 
                                confidences: List) -> Dict:
        """Comprehensive performance analysis"""
        
        # Convert string labels to binary
        y_true = [1 if label == 'medical' else 0 for label in true_labels]
        y_pred = [1 if pred == 'medical' else 0 for pred in predictions]
        y_scores = confidences
        
        analysis = {}
        
        # ROC Curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        analysis['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        }
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        analysis['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'auc': float(pr_auc)
        }
        
        # Calibration Curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_scores, n_bins=10
        )
        
        analysis['calibration'] = {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        }
        
        # Generate interactive plots
        self._create_interactive_plots(analysis)
        
        return analysis
    
    def _create_interactive_plots(self, analysis: Dict):
        """Create interactive performance plots using Plotly"""
        
        # ROC Curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=analysis['roc_curve']['fpr'],
            y=analysis['roc_curve']['tpr'],
            mode='lines',
            name=f'ROC Curve (AUC = {analysis["roc_curve"]["auc"]:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600, height=500
        )
        fig_roc.write_html('roc_curve.html')
        
        # Precision-Recall Curve
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=analysis['pr_curve']['recall'],
            y=analysis['pr_curve']['precision'],
            mode='lines',
            name=f'PR Curve (AUC = {analysis["pr_curve"]["auc"]:.3f})',
            line=dict(color='green', width=2)
        ))
        fig_pr.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600, height=500
        )
        fig_pr.write_html('pr_curve.html')
        
        # Calibration Plot
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=analysis['calibration']['mean_predicted_value'],
            y=analysis['calibration']['fraction_of_positives'],
            mode='markers+lines',
            name='Model Calibration',
            line=dict(color='purple', width=2),
            marker=dict(size=8)
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='red', dash='dash')
        ))
        fig_cal.update_layout(
            title='Calibration Plot',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=600, height=500
        )
        fig_cal.write_html('calibration_plot.html')
        
        print("Interactive plots saved as HTML files:")
        print("- roc_curve.html")
        print("- pr_curve.html") 
        print("- calibration_plot.html")