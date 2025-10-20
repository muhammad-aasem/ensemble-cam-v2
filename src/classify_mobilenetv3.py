#!/usr/bin/env python3
"""
X-ray Image Classifier using MobileNetV3
Predicts pathologies from chest X-ray images using custom trained MobileNetV3.
Input: 224×224 pixels
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchxrayvision as xrv
from PIL import Image
from skimage.io import imread
import skimage

# Set local weights directory for portability
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "models_data"
TRAIN_DIR = PROJECT_ROOT / "weights" / "train_weight_mobilenet"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


class MobileNetV3XRay(nn.Module):
    """
    MobileNetV3 model adapted for chest X-ray classification.
    Compatible with TorchXRayVision's multi-label classification format.
    """
    
    def __init__(self, num_classes=14):
        """
        Initialize MobileNetV3 for X-ray classification.
        
        Args:
            num_classes: Number of pathology classes (NIH has 14)
        """
        super(MobileNetV3XRay, self).__init__()
        
        # Load MobileNetV3 Large
        from torchvision.models import mobilenet_v3_large
        mobilenet = mobilenet_v3_large(pretrained=False)
        
        # Modify first conv layer to accept 1-channel (grayscale) input
        original_conv = mobilenet.features[0][0]
        mobilenet.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
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


def load_model(weights_path=None, device=None):
    """
    Load trained MobileNetV3 model.
    
    Args:
        weights_path: Path to model weights (default: best trained model)
        device: Device to load model on (cuda/cpu)
    
    Returns:
        Loaded model and device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Default to best model if no path specified
    if weights_path is None:
        weights_path = TRAIN_DIR / "mobilenetv3_best.pt"
    else:
        weights_path = Path(weights_path)
    
    if not weights_path.exists():
        print(f"Error: Model weights not found at {weights_path}", file=sys.stderr)
        print(f"\nPlease train the model first using:", file=sys.stderr)
        print(f"  uv run python src/train_mobilenet.py --epochs 50", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading model: {weights_path}")
    print(f"Device: {device}")
    print(f"Architecture: MobileNetV3-Large (224×224 input)")
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Create model
    num_classes = len(checkpoint['pathologies'])
    model = MobileNetV3XRay(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Store pathologies in model
    model.pathologies = checkpoint['pathologies']
    
    print(f"Model loaded successfully!")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best AUC: {checkpoint['best_auc']:.4f}")
    print(f"  Pathologies: {len(model.pathologies)}")
    
    return model, device


def preprocess_image(image_path):
    """
    Preprocess the X-ray image for MobileNetV3 model input (224×224).
    Uses the same preprocessing pipeline as TorchXRayVision.
    
    Args:
        image_path: Path to the X-ray image
    
    Returns:
        Preprocessed image tensor (1, 1, 224, 224)
    """
    # Load image using skimage (as torchxrayvision does)
    img = imread(image_path)
    
    # Convert to grayscale if RGB/RGBA
    if len(img.shape) > 2:
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]  # Drop alpha
        # Convert RGB to grayscale
        img = img.mean(axis=2)
    
    # Convert to float32
    img = img.astype(np.float32)
    
    # Normalize to [0, 255] range if needed
    if img.max() <= 1.0:
        img = img * 255.0
    
    # Resize using skimage to 224×224
    from skimage.transform import resize
    img = resize(img, (224, 224), anti_aliasing=True, preserve_range=True)
    
    # Normalize pixel values using torchxrayvision's normalize function
    img = xrv.datasets.normalize(img, maxval=255, reshape=False)
    
    # Ensure we have the right shape (224, 224)
    if img.shape != (224, 224):
        raise ValueError(f"Image shape is {img.shape}, expected (224, 224)")
    
    # Add batch and channel dimensions
    # Shape: (1, 1, 224, 224)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    
    return img


def predict(model, image_tensor, device):
    """
    Generate predictions for the image.
    
    Args:
        model: Trained MobileNetV3 model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
    
    Returns:
        Dictionary mapping pathology names to probability scores
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply sigmoid to get probabilities (multi-label classification)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Create results dictionary
    results = {
        pathology: float(prob) 
        for pathology, prob in zip(model.pathologies, probabilities)
    }
    
    return results


def print_predictions(results, threshold=0.5):
    """
    Print formatted prediction results.
    
    Args:
        results: Dictionary of pathology predictions
        threshold: Threshold for positive classification
    """
    print("\n" + "="*60)
    print("X-RAY CLASSIFICATION RESULTS (MobileNetV3)")
    print("="*60)
    
    # Sort by probability (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nAll Predictions (sorted by probability):")
    print("-"*60)
    for pathology, prob in sorted_results:
        status = "POSITIVE" if prob >= threshold else "negative"
        print(f"{pathology:30s}: {prob:.4f} ({status})")
    
    # Print positive findings
    positive_findings = [(p, prob) for p, prob in sorted_results if prob >= threshold]
    
    print("\n" + "="*60)
    if positive_findings:
        print(f"\nPositive Findings (≥{threshold}):")
        for pathology, prob in positive_findings:
            print(f"  • {pathology}: {prob:.4f}")
    else:
        print(f"\nNo positive findings (threshold: {threshold})")
    
    print("="*60 + "\n")


def main():
    """Main function to run the classifier."""
    parser = argparse.ArgumentParser(
        description="Classify chest X-ray images using trained MobileNetV3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python src/classify_mobilenetv3.py --input xray_image.jpg
  python src/classify_mobilenetv3.py --input xray.png --threshold 0.6
  python src/classify_mobilenetv3.py --input xray.jpg --weights weights/train_weight_mobilenet/mobilenetv3_best.pt
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input X-ray image'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold for positive prediction (default: 0.5)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to run inference on (default: auto)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to model weights (default: weights/train_weight_mobilenet/mobilenetv3_best.pt)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    image_path = Path(args.input)
    if not image_path.exists():
        print(f"Error: Image file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    try:
        # Load model
        model, device = load_model(weights_path=args.weights, device=device)
        
        # Preprocess image
        print(f"Processing image: {args.input}")
        image_tensor = preprocess_image(args.input)
        
        # Generate predictions
        print("Generating predictions...")
        results = predict(model, image_tensor, device)
        
        # Print results
        print_predictions(results, threshold=args.threshold)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

