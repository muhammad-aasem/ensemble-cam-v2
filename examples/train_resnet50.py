#!/usr/bin/env python3
"""
ResNet-50 Training Script

This script takes a directory of preprocessed dataset and performs training
using ResNet-50 architecture.

Usage:
    python train_resnet50.py --data_dir ./data/processed --output_dir ./outputs/resnet50_training
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class ProcessedDataset(Dataset):
    """Dataset class for preprocessed data."""
    
    def __init__(self, data_dir: str, split: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing processed data
            split: Data split ('train', 'validation', 'test')
            transform: Optional transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load metadata
        csv_path = self.data_dir / split / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.data = pd.read_csv(csv_path)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        row = self.data.iloc[idx]
        
        # Load preprocessed image tensor
        image_path = row['processed_image_path']
        image_tensor = torch.load(image_path)
        
        # Get label (assuming 'label' column exists)
        label = row.get('label', 0)
        if isinstance(label, str):
            # Convert string labels to integers
            unique_labels = sorted(self.data['label'].unique())
            label = unique_labels.index(label)
        
        return {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'index': idx
        }


class ResNet50Classifier(nn.Module):
    """ResNet-50 classifier for image classification."""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout_rate: float = 0.5):
        """
        Initialize ResNet-50 classifier.
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for the classifier
        """
        super(ResNet50Classifier, self).__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace the final classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)


class Trainer:
    """Training class for ResNet-50."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        output_dir: str,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        patience: int = 5
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to run training on
            output_dir: Output directory for saving models and logs
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            patience: Early stopping patience
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.logger = logging.getLogger(__name__)
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%'
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def test_model(self) -> Dict[str, float]:
        """Test the model and return metrics."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Generate classification report
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        return metrics, report
    
    def save_model(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'model_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with validation accuracy: {self.best_val_acc:.4f}")
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.num_epochs} epochs...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        
        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log progress
            self.logger.info(
                f'Epoch {epoch}/{self.num_epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            )
            
            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save model
            self.save_model(epoch, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        self.logger.info("Training completed!")
        
        # Final evaluation
        self.logger.info("Evaluating on test set...")
        metrics, report = self.test_model()
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")
        self.logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Save final results
        results = {
            'final_metrics': metrics,
            'classification_report': report,
            'training_history': self.history,
            'best_val_acc': self.best_val_acc
        }
        
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Directory containing processed data
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger = logging.getLogger(__name__)
    
    # Create datasets
    train_dataset = ProcessedDataset(data_dir, 'train')
    val_dataset = ProcessedDataset(data_dir, 'validation')
    test_dataset = ProcessedDataset(data_dir, 'test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train ResNet-50 model")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of output classes (default: 2)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use ('cpu', 'cuda', or 'auto')"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Create model
        logger.info(f"Creating ResNet-50 model with {args.num_classes} classes...")
        model = ResNet50Classifier(num_classes=args.num_classes)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            patience=args.patience
        )
        
        # Train model
        results = trainer.train()
        
        logger.info(f"🎉 Training completed successfully!")
        logger.info(f"📁 Results saved to: {args.output_dir}")
        logger.info(f"🏆 Best validation accuracy: {results['best_val_acc']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
