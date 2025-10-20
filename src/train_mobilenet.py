#!/usr/bin/env python3
"""
MobileNetV3 Training Script for NIH Chest X-ray14 Dataset
Compatible with TorchXRayVision infrastructure for chest X-ray classification.

Usage:
    uv run python src/train_mobilenet.py --epochs 50
    uv run python src/train_mobilenet.py --epochs 100 --batch-size 32 --lr 0.001
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchxrayvision as xrv
from tqdm import tqdm

# Set local weights directory
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "models_data"
TRAIN_DIR = PROJECT_ROOT / "weights" / "train_weight_mobilenet"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# Performance CSV path
PERFORMANCE_CSV = TRAIN_DIR / "performance.csv"
BEST_MODEL_PATH = TRAIN_DIR / "mobilenetv3_best.pt"
LAST_MODEL_PATH = TRAIN_DIR / "mobilenetv3_last.pt"


class MobileNetV3XRay(nn.Module):
    """
    MobileNetV3 model adapted for chest X-ray classification.
    Compatible with TorchXRayVision's multi-label classification format.
    """
    
    def __init__(self, num_classes=14, pretrained=True):
        """
        Initialize MobileNetV3 for X-ray classification.
        
        Args:
            num_classes: Number of pathology classes (NIH has 14)
            pretrained: Use ImageNet pretrained weights
        """
        super(MobileNetV3XRay, self).__init__()
        
        # Load MobileNetV3 Large
        from torchvision.models import mobilenet_v3_large
        mobilenet = mobilenet_v3_large(pretrained=pretrained)
        
        # Modify first conv layer to accept 1-channel (grayscale) input
        original_conv = mobilenet.features[0][0]
        mobilenet.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Initialize the new conv layer weights by averaging RGB weights
        if pretrained:
            with torch.no_grad():
                mobilenet.features[0][0].weight.copy_(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # Extract feature extractor (remove classifier)
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        
        # Custom classifier for multi-label classification
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )
        
        # Store pathology labels (NIH Chest X-ray14)
        self.pathologies = [
            "Atelectasis", "Consolidation", "Infiltration",
            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
            "Effusion", "Pneumonia", "Pleural_Thickening",
            "Cardiomegaly", "Nodule", "Mass", "Hernia"
        ]
        
    def forward(self, x):
        """Forward pass through the network."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_datasets(imgpath, csvpath, batch_size=16, num_workers=4):
    """
    Load NIH Chest X-ray14 dataset using TorchXRayVision.
    
    Args:
        imgpath: Path to NIH images directory
        csvpath: Path to Data_Entry_2017.csv
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, num_classes
    """
    print("Loading NIH Chest X-ray14 dataset...")
    
    # Define transforms for data augmentation
    train_transforms = xrv.datasets.XRayResizer(224)
    
    # Load NIH dataset
    try:
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=imgpath,
            csvpath=csvpath,
            transform=train_transforms,
            unique_patients=False
        )
    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR: Could not load NIH Chest X-ray14 dataset")
        print(f"{'='*60}")
        print("\nPlease download the NIH Chest X-ray14 dataset:")
        print("1. Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC")
        print("2. Extract images to a directory")
        print("3. Provide the paths using command-line arguments:")
        print(f"   --imgpath: Path to images directory")
        print(f"   --csvpath: Path to Data_Entry_2017.csv")
        print(f"\nExample:")
        print(f"  uv run python src/train_mobilenet.py --epochs 50 \\")
        print(f"    --imgpath /path/to/images \\")
        print(f"    --csvpath /path/to/Data_Entry_2017.csv")
        print(f"\nOriginal error: {e}")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    # Split into train/val (80/20)
    num_samples = len(dataset)
    indices = list(range(num_samples))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(dataset.pathologies)
    
    print(f"Dataset loaded successfully!")
    print(f"  Total samples: {num_samples}")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Pathologies: {dataset.pathologies}")
    
    return train_loader, val_loader, num_classes


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Get images and labels
        images = batch["img"].to(device)
        labels = batch["lab"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss (multi-label classification)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = running_loss / len(train_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate AUC for each class
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro')
    except:
        auc = 0.0
    
    return avg_loss, auc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ")
        for batch in pbar:
            images = batch["img"].to(device)
            labels = batch["lab"].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = running_loss / len(val_loader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro')
    except:
        auc = 0.0
    
    return avg_loss, auc


def save_checkpoint(model, optimizer, epoch, best_auc, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
        'pathologies': model.pathologies,
    }
    
    # Always save last checkpoint
    torch.save(checkpoint, LAST_MODEL_PATH)
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, BEST_MODEL_PATH)
        print(f"✓ Saved best model to {BEST_MODEL_PATH}")


def load_checkpoint(model, optimizer=None):
    """Load the last checkpoint if it exists."""
    if BEST_MODEL_PATH.exists():
        print(f"Loading checkpoint from {BEST_MODEL_PATH}")
        checkpoint = torch.load(BEST_MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint['best_auc']
        
        print(f"Resuming from epoch {start_epoch}, best AUC: {best_auc:.4f}")
        return start_epoch, best_auc
    
    return 0, 0.0


def init_performance_csv():
    """Initialize performance CSV file with headers."""
    if not PERFORMANCE_CSV.exists():
        with open(PERFORMANCE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_auc', 'val_loss', 'val_auc',
                'learning_rate', 'timestamp'
            ])


def log_performance(epoch, train_loss, train_auc, val_loss, val_auc, lr):
    """Log performance metrics to CSV."""
    with open(PERFORMANCE_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, f'{train_loss:.6f}', f'{train_auc:.6f}',
            f'{val_loss:.6f}', f'{val_auc:.6f}', f'{lr:.8f}',
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3 on NIH Chest X-ray14 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  uv run python src/train_mobilenet.py --epochs 50 --imgpath /path/to/images --csvpath /path/to/Data_Entry_2017.csv
  uv run python src/train_mobilenet.py --epochs 100 --batch-size 32 --lr 0.001 --imgpath /path/to/images --csvpath /path/to/csv
  uv run python src/train_mobilenet.py --epochs 200 --device cuda --imgpath /path/to/images --csvpath /path/to/csv
        """
    )
    
    parser.add_argument(
        '--imgpath',
        type=str,
        required=True,
        help='Path to NIH Chest X-ray14 images directory'
    )
    
    parser.add_argument(
        '--csvpath',
        type=str,
        required=True,
        help='Path to Data_Entry_2017.csv file'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs to train (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to train on (default: auto)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print("MobileNetV3 Training for NIH Chest X-ray14")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Training directory: {TRAIN_DIR}")
    print(f"Performance CSV: {PERFORMANCE_CSV}")
    print(f"{'='*60}\n")
    
    # Initialize performance CSV
    init_performance_csv()
    
    # Validate dataset paths
    if not Path(args.imgpath).exists():
        print(f"Error: Image directory not found: {args.imgpath}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.csvpath).exists():
        print(f"Error: CSV file not found: {args.csvpath}", file=sys.stderr)
        sys.exit(1)
    
    # Load datasets
    train_loader, val_loader, num_classes = load_datasets(
        imgpath=args.imgpath,
        csvpath=args.csvpath,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\nInitializing MobileNetV3 model...")
    model = MobileNetV3XRay(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    print(f"Model architecture: MobileNetV3-Large")
    print(f"Input size: 224x224")
    print(f"Number of classes: {num_classes}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Load checkpoint if exists
    start_epoch, best_auc = load_checkpoint(model, optimizer)
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    prev_lr = args.lr
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        
        # Validate
        val_loss, val_auc = validate(
            model, val_loader, criterion, device, epoch + 1
        )
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check if learning rate changed
        if current_lr != prev_lr:
            print(f"\n  📉 Learning rate reduced: {prev_lr:.8f} → {current_lr:.8f}")
            prev_lr = current_lr
        
        # Log performance
        log_performance(
            epoch + 1, train_loss, train_auc, val_loss, val_auc, current_lr
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")
        print(f"  Learning Rate: {current_lr:.8f}")
        
        # Save checkpoint
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            print(f"  🎯 New best validation AUC: {best_auc:.4f}")
        
        save_checkpoint(model, optimizer, epoch, best_auc, is_best)
        print(f"{'='*60}\n")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Last model saved to: {LAST_MODEL_PATH}")
    print(f"Performance log: {PERFORMANCE_CSV}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

