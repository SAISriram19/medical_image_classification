"""
Self-supervised training module for medical image classification
Implements contrastive learning and advanced augmentation strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import random
from typing import List, Tuple, Dict
import logging
from pathlib import Path
import json
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MedicalImageDataset(Dataset):
    """Advanced dataset class with sophisticated augmentations"""
    
    def __init__(self, images: List[Tuple], transform=None, is_training=True):
        self.images = images
        self.transform = transform
        self.is_training = is_training
        
        # Advanced medical-specific augmentations
        self.medical_augment = A.Compose([
            # Geometric transformations
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            
            # Intensity transformations (medical-specific)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # Noise and blur (simulating acquisition artifacts)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            
            # Medical-specific augmentations
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.2),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.test_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image, label, _ = self.images[idx]
        
        # Convert PIL to numpy for albumentations
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Apply augmentations
        if self.is_training:
            augmented = self.medical_augment(image=image)
            image = augmented['image']
        else:
            augmented = self.test_transform(image=image)
            image = augmented['image']
        
        # Convert label to tensor
        label_tensor = torch.tensor(1 if label == 'medical' else 0, dtype=torch.long)
        
        return image, label_tensor

class ContrastiveLoss(nn.Module):
    """Contrastive loss for self-supervised learning"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive and negative masks
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size).to(features.device)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
        
        log_prob = similarity_matrix - torch.log(sum_exp_sim)
        mean_log_prob_pos = torch.sum(mask * log_prob, dim=1) / torch.sum(mask, dim=1)
        
        loss = -mean_log_prob_pos.mean()
        return loss

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class AdvancedTrainer:
    """Advanced training pipeline with multiple loss functions and techniques"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Multiple loss functions
        self.classification_loss = FocalLoss(alpha=1, gamma=2)
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
        # Optimizers with different learning rates
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Classification loss
            cls_loss = self.classification_loss(outputs, targets)
            
            # Feature extraction for contrastive loss
            with torch.no_grad():
                features = self.model.features(data)
                features = self.model.adaptive_pool(features)
                features = torch.flatten(features, 1)
            
            # Contrastive loss
            cont_loss = self.contrastive_loss(features, targets)
            
            # Combined loss
            total_batch_loss = cls_loss + 0.1 * cont_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += total_batch_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.classification_loss(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, epochs=50, save_path='best_model.pth'):
        """Full training loop"""
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        # Initialize wandb for experiment tracking
        wandb.init(project="medical-image-classification", 
                  config={
                      "epochs": epochs,
                      "batch_size": train_loader.batch_size,
                      "learning_rate": self.optimizer.param_groups[0]['lr'],
                      "model": "AdvancedEnsemble"
                  })
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'learning_rate': current_lr
            })
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:'
                  f' Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%'
                  f' Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
                  f' LR: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, save_path)
                patience_counter = 0
                print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {patience} epochs without improvement')
                break
        
        # Generate final report
        self.generate_training_report(val_preds, val_targets)
        wandb.finish()
        
        return self.history
    
    def generate_training_report(self, predictions, targets):
        """Generate comprehensive training report"""
        # Classification report
        class_names = ['Non-Medical', 'Medical']
        report = classification_report(targets, predictions, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        print("\n=== Classification Report ===")
        print(classification_report(targets, predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training curves
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(self.history['learning_rates'])
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax4)
        ax4.set_title('Confusion Matrix')
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('training_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed report
        detailed_report = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'training_history': self.history,
            'final_metrics': {
                'best_val_accuracy': max(self.history['val_acc']),
                'final_train_accuracy': self.history['train_acc'][-1],
                'final_val_accuracy': self.history['val_acc'][-1]
            }
        }
        
        with open('detailed_training_report.json', 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print("Training report saved to 'training_report.png' and 'detailed_training_report.json'")

def create_synthetic_training_data(num_samples=1000):
    """Create synthetic training data for demonstration"""
    from src.advanced_models import AdvancedEnsembleClassifier
    
    # This would normally load real medical datasets
    # For demo purposes, we'll create synthetic data
    
    print(f"Creating {num_samples} synthetic training samples...")
    
    # Implementation would go here to create diverse synthetic medical/non-medical images
    # This is a placeholder for the actual implementation
    
    return []  # Return list of (image, label, metadata) tuples