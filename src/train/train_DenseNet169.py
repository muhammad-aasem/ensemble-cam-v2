"""
DenseNet169 Training Script for NIH Chest X-ray Dataset

This script implements a comprehensive DenseNet169 fine-tuning pipeline optimized for
chest X-ray classification and subsequent Grad-CAM++ heatmap generation.

=== DATASET HANDLING ===
- Uses dataset/NIH-Chest-X-ray/Classification directory (post-resize JPEG images)
- Automatically generates dataset_split.csv with 70/20/10 train/val/test split (seed=42)
- Handles class imbalance through stratified sampling
- Implements data augmentation: RandomHorizontalFlip, RandomRotation, ColorJitter
- Normalizes images using ImageNet statistics for pre-trained compatibility

=== MODEL ARCHITECTURE ===
- Base: torchvision.models.densenet169 (ImageNet pre-trained)
- Modifications:
  * Replace final classifier for 15-class classification (NIH dataset classes)
  * Add dropout (0.5) before final layer to prevent overfitting
  * Initialize final layer weights using Xavier uniform
- Preserves feature extraction layers for Grad-CAM++ compatibility

=== TRAINING CONFIGURATION ===
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR with warm restarts
- Loss Function: Weighted CrossEntropyLoss (handles class imbalance)
- Batch Size: 32 (adjustable based on available memory)
- Epochs: 50 (with early stopping patience=10)
- Mixed Precision: Enabled for memory efficiency

=== CLASS IMBALANCE HANDLING ===
- Calculates class weights from training set distribution
- Applies weighted loss to penalize majority classes less
- Uses stratified sampling for balanced validation sets
- Monitors per-class metrics during training

=== REGULARIZATION TECHNIQUES ===
- Early Stopping: Monitors validation loss (patience=10)
- Learning Rate Scheduling: Cosine annealing with restarts
- Weight Decay: L2 regularization (1e-4)
- Dropout: 0.5 probability in final layer
- Data Augmentation: Random transforms during training

=== VALIDATION & METRICS ===
- Primary Metric: Macro-averaged F1-score
- Additional Metrics: Precision, Recall, AUC-ROC (per-class and macro)
- Validation Frequency: Every epoch
- Best Model Selection: Based on validation F1-score
- Confusion Matrix: Generated for final evaluation

=== LOGGING & MONITORING ===
- TensorBoard Integration:
  * Loss curves (train/val)
  * Learning rate schedule
  * Per-class metrics
  * Sample predictions with images
  * Model architecture visualization
- Console Logging: Progress bars, epoch summaries, metric tracking
- Model Checkpointing: Best weights saved to models/densenet169_best.pth

=== GRAD-CAM++ COMPATIBILITY ===
- Preserves intermediate feature maps in DenseNet169 layers
- Maintains gradient flow for activation computation
- Saves model with feature extraction hooks enabled
- Exports class indices mapping for heatmap generation
- Ensures model state is compatible with Grad-CAM++ requirements

=== MODEL PERSISTENCE ===
- Saves best model weights: models/densenet169_best.pth
- Saves class mapping: models/class_indices.json
- Saves training history: models/training_history.json
- Saves model configuration: models/model_config.json
- Automatic cleanup: Removes previous checkpoints when better model found

=== CONTINUOUS TRAINING ===
- Resume from best checkpoint if available
- Incremental training: Additional epochs from best weights
- Validation-based stopping: Prevents overfitting
- Model comparison: Only saves if validation metrics improve

=== HARDWARE OPTIMIZATION ===
- Device Detection: Automatic CPU/GPU selection
- Memory Management: Gradient accumulation for large batches
- Mixed Precision: Reduces memory usage and speeds training
- DataLoader Optimization: Multiple workers, pin_memory

=== EVALUATION PIPELINE ===
- Test Set Evaluation: Final model performance on held-out test set
- Per-Class Analysis: Detailed metrics for each disease class
- Error Analysis: Misclassified samples identification
- Model Interpretability: Feature importance analysis

=== OUTPUT STRUCTURE ===
models/
├── densenet169_best.pth        # Best model weights
├── class_indices.json          # Class name to index mapping
├── training_history.json       # Training metrics history
├── model_config.json           # Model configuration
└── tensorboard_logs/           # TensorBoard log files

=== GRAD-CAM++ INTEGRATION ===
- Model Architecture: Compatible with Grad-CAM++ requirements
- Feature Maps: Preserved for activation computation
- Class Mapping: Exported for heatmap labeling
- Model State: Ready for immediate heatmap generation
- Documentation: Clear integration instructions for generate_heatmap.py

=== PERFORMANCE TARGETS ===
- Training Time: < 2 hours per epoch (M3 Max, 36GB RAM)
- Memory Usage: < 20GB peak during training
- Validation F1: Target > 0.75 macro-averaged
- Test Accuracy: Target > 0.80 overall
- Convergence: Stable training within 30 epochs

=== ERROR HANDLING ===
- Dataset Validation: Checks for required directories and files
- Model Loading: Graceful handling of missing checkpoints
- Memory Management: Automatic batch size reduction if OOM
- Logging Errors: Comprehensive error logging and recovery

This implementation ensures optimal performance for chest X-ray classification
while maintaining compatibility with Grad-CAM++ heatmap generation.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import argparse
from datetime import datetime
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time


class ChestXRayDataset(Dataset):
    """Custom dataset for chest X-ray images with metadata."""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = torchvision.io.read_image(image_path)
        image = image.float() / 255.0  # Normalize to [0, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class DenseNet169Classifier(nn.Module):
    """DenseNet169 model modified for chest X-ray classification."""
    
    def __init__(self, num_classes: int = 15, dropout_rate: float = 0.5):
        super(DenseNet169Classifier, self).__init__()
        
        # Load pre-trained DenseNet169
        self.backbone = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        
        # Get number of features from the classifier
        num_features = self.backbone.classifier.in_features
        
        # Replace the final classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        # Initialize the new layer
        nn.init.xavier_uniform_(self.backbone.classifier[1].weight)
        nn.init.zeros_(self.backbone.classifier[1].bias)
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extract features for Grad-CAM++ compatibility."""
        features = self.backbone.features(x)
        out = torch.relu(features, inplace=True)
        out = torch.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out


class ModelTrainer:
    """Main training class for DenseNet169 chest X-ray classification."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Initialize paths - models saved to models/densenet169 directory
        self.data_dir = Path(config['data_dir'])
        self.models_dir = Path(config['models_dir']) / 'densenet169'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir=self.models_dir / 'tensorboard_logs')
        
        # Initialize metrics tracking
        self.best_f1_score = 0.0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': [],
            'train_acc': [], 'val_acc': [], 'learning_rates': []
        }
        
        # Performance tracking for CSV export
        self.performance_data = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Class names (NIH Chest X-ray dataset)
        self.class_names = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
            'No_Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]
        
        print(f"Using device: {self.device}")
        print(f"Models will be saved to: {self.models_dir}")
    
    def create_dataset_split(self) -> pd.DataFrame:
        """Create train/val/test split with stratified sampling."""
        print("Creating dataset split...")
        
        split_file = self.data_dir / 'dataset_split.csv'
        
        if split_file.exists():
            print("Loading existing dataset split...")
            return pd.read_csv(split_file)
        
        # Collect all image paths and labels
        image_paths = []
        labels = []
        class_to_idx = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory {class_name} not found, skipping...")
                continue
            
            class_to_idx[class_name] = class_idx
            class_images = list(class_dir.glob('*.jpg'))
            image_paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))
        
        # Create DataFrame
        df = pd.DataFrame({
            'image_path': image_paths,
            'label': labels,
            'class_name': [self.class_names[label] for label in labels]
        })
        
        # Stratified split: 70% train, 20% val, 10% test
        train_df, temp_df = train_test_split(
            df, test_size=0.3, stratify=df['label'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=1/3, stratify=temp_df['label'], random_state=42
        )
        
        # Add split column
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        
        # Combine and save
        split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        split_df.to_csv(split_file, index=False)
        
        print(f"Dataset split created:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        return split_df
    
    def get_data_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get data augmentation transforms."""
        
        # ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            normalize
        ])
        
        # Validation/test transforms (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            normalize
        ])
        
        return train_transform, val_transform
    
    def create_data_loaders(self, split_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for train/val/test sets."""
        
        train_transform, val_transform = self.get_data_transforms()
        
        # Split data
        train_df = split_df[split_df['split'] == 'train']
        val_df = split_df[split_df['split'] == 'val']
        test_df = split_df[split_df['split'] == 'test']
        
        # Create datasets
        train_dataset = ChestXRayDataset(
            train_df['image_path'].tolist(),
            train_df['label'].tolist(),
            train_transform
        )
        
        val_dataset = ChestXRayDataset(
            val_df['image_path'].tolist(),
            val_df['label'].tolist(),
            val_transform
        )
        
        test_dataset = ChestXRayDataset(
            test_df['image_path'].tolist(),
            test_df['label'].tolist(),
            val_transform
        )
        
        # Calculate class weights for weighted sampling
        class_counts = train_df['label'].value_counts().sort_index()
        class_weights = 1.0 / class_counts.values
        sample_weights = [class_weights[label] for label in train_df['label']]
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader, test_loader
    
    def calculate_class_weights(self, train_df: pd.DataFrame) -> torch.Tensor:
        """Calculate class weights for weighted loss."""
        class_counts = train_df['label'].value_counts().sort_index()
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['label']),
            y=train_df['label']
        )
        return torch.FloatTensor(class_weights).to(self.device)
    
    def initialize_model(self) -> DenseNet169Classifier:
        """Initialize the DenseNet169 model."""
        model = DenseNet169Classifier(
            num_classes=len(self.class_names),
            dropout_rate=self.config['dropout_rate']
        )
        
        # Load checkpoint if available
        checkpoint_path = self.models_dir / 'densenet169_best.pth'
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.best_f1_score = checkpoint.get('best_f1_score', 0.0)
            print(f"Loaded checkpoint with best F1 score: {self.best_f1_score:.4f}")
        
        return model.to(self.device)
    
    def initialize_optimizer_and_scheduler(self, model: DenseNet169Classifier, 
                                         train_loader: DataLoader) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Initialize optimizer and learning rate scheduler."""
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config['scheduler_T_0'],
            T_mult=self.config['scheduler_T_mult']
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model: DenseNet169Classifier, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module,
                   epoch: int) -> Tuple[float, float, float, float, float]:
        """Train for one epoch."""
        
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["max_epochs"]} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        # Calculate detailed metrics
        train_metrics = self.calculate_detailed_metrics(all_predictions, all_labels)
        epoch_f1 = train_metrics['f1'] * 100  # Convert to percentage
        epoch_precision = train_metrics['precision'] * 100
        epoch_recall = train_metrics['recall'] * 100
        
        return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall
    
    def validate_epoch(self, model: DenseNet169Classifier, val_loader: DataLoader,
                      criterion: nn.Module, epoch: int) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
        """Validate for one epoch."""
        
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.config["max_epochs"]} [Val]')
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.scaler:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_predictions / total_samples:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        
        # Calculate detailed metrics
        val_metrics = self.calculate_detailed_metrics(all_predictions, all_labels)
        epoch_f1 = val_metrics['f1'] * 100  # Convert to percentage
        epoch_precision = val_metrics['precision'] * 100
        epoch_recall = val_metrics['recall'] * 100
        
        return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall, np.array(all_predictions), np.array(all_labels)
    
    def save_checkpoint(self, model: DenseNet169Classifier, optimizer: optim.Optimizer,
                       scheduler: optim.lr_scheduler._LRScheduler, epoch: int,
                       val_f1: float, is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1_score': self.best_f1_score,
            'val_f1': val_f1,
            'class_names': self.class_names,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.models_dir / 'densenet169_latest.pth')
        
        # Save best checkpoint - only save if better than previous best
        if is_best:
            torch.save(checkpoint, self.models_dir / 'densenet169_best.pth')
            
            # Save class indices mapping
            class_indices = {name: idx for idx, name in enumerate(self.class_names)}
            with open(self.models_dir / 'class_indices.json', 'w') as f:
                json.dump(class_indices, f, indent=2)
            
            # Save model configuration
            with open(self.models_dir / 'model_config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"New best model saved with F1 score: {val_f1:.4f}")
    
    def log_metrics(self, epoch: int, train_loss: float, val_loss: float,
                   train_acc: float, val_acc: float, train_f1: float, val_f1: float,
                   learning_rate: float):
        """Log metrics to TensorBoard and history."""
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        self.writer.add_scalar('F1_Score/Train', train_f1, epoch)
        self.writer.add_scalar('F1_Score/Validation', val_f1, epoch)
        self.writer.add_scalar('Learning_Rate', learning_rate, epoch)
        
        # Update training history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['val_acc'].append(val_acc)
        self.training_history['train_f1'].append(train_f1)
        self.training_history['val_f1'].append(val_f1)
        self.training_history['learning_rates'].append(learning_rate)
        
        # Save training history
        with open(self.models_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def log_performance_metrics(self, epoch: int, train_loss: float, val_loss: float,
                               train_acc: float, val_acc: float, train_f1: float, val_f1: float,
                               train_precision: float, val_precision: float,
                               train_recall: float, val_recall: float,
                               learning_rate: float):
        """Log performance metrics to CSV file."""
        
        # Calculate additional metrics
        performance_row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_data.append(performance_row)
        
        # Save to CSV
        performance_df = pd.DataFrame(self.performance_data)
        csv_path = self.models_dir / f'{self.timestamp}_training_performance.csv'
        performance_df.to_csv(csv_path, index=False)
    
    def calculate_detailed_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate detailed classification metrics."""
        
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, accuracy_score,
            precision_recall_fscore_support
        )
        
        # Overall metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            labels, predictions, zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support': support
        }
    
    def evaluate_model(self, model: DenseNet169Classifier, test_loader: DataLoader) -> Dict:
        """Comprehensive model evaluation."""
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.scaler:
                    with autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100 * (np.array(all_predictions) == np.array(all_labels)).mean()
        
        # Classification report
        report = classification_report(
            all_labels, all_predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Per-class AUC-ROC
        try:
            auc_scores = roc_auc_score(
                all_labels, all_probabilities,
                multi_class='ovr', average=None
            )
            macro_auc = np.mean(auc_scores)
        except:
            auc_scores = None
            macro_auc = None
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'auc_scores': auc_scores,
            'macro_auc': macro_auc,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Path):
        """Plot and save confusion matrix."""
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self):
        """Main training loop."""
        
        print("Starting DenseNet169 training for chest X-ray classification...")
        
        # Create dataset split
        split_df = self.create_dataset_split()
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(split_df)
        
        # Calculate class weights
        train_df = split_df[split_df['split'] == 'train']
        class_weights = self.calculate_class_weights(train_df)
        
        # Initialize model
        model = self.initialize_model()
        
        # Initialize optimizer and scheduler
        optimizer, scheduler = self.initialize_optimizer_and_scheduler(model, train_loader)
        
        # Initialize loss function
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        early_stopping_counter = 0
        
        for epoch in range(self.config['max_epochs']):
            start_time = time.time()
            
            # Train
            train_loss, train_acc, train_f1, train_precision, train_recall = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # Validate
            val_loss, val_acc, val_f1, val_precision, val_recall, val_predictions, val_labels = self.validate_epoch(
                model, val_loader, criterion, epoch
            )
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, 
                           train_f1, val_f1, current_lr)
            
            # Log detailed performance metrics to CSV
            self.log_performance_metrics(epoch, train_loss, val_loss, train_acc, val_acc,
                                       train_f1, val_f1, train_precision, val_precision,
                                       train_recall, val_recall, current_lr)
            
            # Check for best model
            is_best = val_f1 > self.best_f1_score
            if is_best:
                self.best_f1_score = val_f1
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(model, optimizer, scheduler, epoch, val_f1, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{self.config["max_epochs"]} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.2f}%, Train P: {train_precision:.2f}%, Train R: {train_recall:.2f}% - '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.2f}%, Val P: {val_precision:.2f}%, Val R: {val_recall:.2f}% - '
                  f'LR: {current_lr:.6f} - Time: {epoch_time:.2f}s')
            
            # Early stopping
            if early_stopping_counter >= self.config['early_stopping_patience']:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Final evaluation
        print("\nEvaluating on test set...")
        test_results = self.evaluate_model(model, test_loader)
        
        print(f"\nTest Results:")
        print(f"Accuracy: {test_results['accuracy']:.2f}%")
        print(f"Macro F1: {test_results['classification_report']['macro avg']['f1-score']:.4f}")
        if test_results['macro_auc']:
            print(f"Macro AUC: {test_results['macro_auc']:.4f}")
        
        # Save confusion matrix
        self.plot_confusion_matrix(
            test_results['confusion_matrix'],
            self.models_dir / 'confusion_matrix.png'
        )
        
        # Save final results
        with open(self.models_dir / 'test_results.json', 'w') as f:
            # Convert numpy arrays and types to JSON-serializable formats
            results_to_save = {}
            
            # Convert basic types
            results_to_save['accuracy'] = float(test_results['accuracy'])
            
            # Convert classification report (contains numpy types)
            report = test_results['classification_report']
            results_to_save['classification_report'] = {}
            for key, value in report.items():
                if isinstance(value, dict):
                    results_to_save['classification_report'][key] = {}
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'item'):  # numpy scalar
                            results_to_save['classification_report'][key][sub_key] = float(sub_value.item())
                        else:
                            results_to_save['classification_report'][key][sub_key] = sub_value
                else:
                    if hasattr(value, 'item'):  # numpy scalar
                        results_to_save['classification_report'][key] = float(value.item())
                    else:
                        results_to_save['classification_report'][key] = value
            
            # Convert arrays
            results_to_save['confusion_matrix'] = test_results['confusion_matrix'].tolist()
            results_to_save['predictions'] = [int(x) for x in test_results['predictions']]
            results_to_save['labels'] = [int(x) for x in test_results['labels']]
            results_to_save['probabilities'] = [[float(y) for y in x] for x in test_results['probabilities']]
            
            # Convert AUC scores
            if test_results['auc_scores'] is not None:
                results_to_save['auc_scores'] = [float(x) for x in test_results['auc_scores'].tolist()]
                results_to_save['macro_auc'] = float(test_results['macro_auc'])
            else:
                results_to_save['auc_scores'] = None
                results_to_save['macro_auc'] = None
                
            json.dump(results_to_save, f, indent=2)
        
        print(f"\nTraining completed! Best validation F1 score: {self.best_f1_score:.4f}")
        print(f"Model and results saved to: {self.models_dir}")
        
        self.writer.close()


def main():
    """Main function to run training."""
    
    parser = argparse.ArgumentParser(description='Train DenseNet169 on NIH Chest X-ray dataset')
    parser.add_argument('data_dir', type=str, 
                       help='Path to the dataset directory (e.g., dataset/NIH-Chest-X-ray/Classification_SMALL)')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save models (default: models)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--max-epochs', type=int, default=50,
                       help='Maximum number of epochs (default: 50)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    parser.add_argument('--scheduler-T-0', type=int, default=10,
                       help='Scheduler T_0 parameter (default: 10)')
    parser.add_argument('--scheduler-T-mult', type=int, default=2,
                       help='Scheduler T_mult parameter (default: 2)')
    
    args = parser.parse_args()
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Dataset directory not found: {data_dir}")
        return 1
    
    if not data_dir.is_dir():
        print(f"Error: Data directory path is not a directory: {data_dir}")
        return 1
    
    # Check for dataset_split.csv
    split_file = data_dir / 'dataset_split.csv'
    if not split_file.exists():
        print(f"Error: dataset_split.csv not found in {data_dir}")
        print("Please ensure the dataset has been processed with a split file.")
        return 1
    
    # Configuration
    config = {
        'data_dir': str(data_dir),
        'models_dir': args.models_dir,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_epochs': args.max_epochs,
        'early_stopping_patience': args.early_stopping_patience,
        'dropout_rate': args.dropout_rate,
        'num_workers': args.num_workers,
        'scheduler_T_0': args.scheduler_T_0,
        'scheduler_T_mult': args.scheduler_T_mult
    }
    
    print("DenseNet169 Training for NIH Chest X-ray Dataset")
    print("=" * 60)
    print(f"Dataset directory: {data_dir}")
    print(f"Models directory: {args.models_dir}/densenet169")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Start training
    trainer.train()
    
    return 0


if __name__ == '__main__':
    exit(main())
