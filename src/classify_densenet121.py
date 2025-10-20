#!/usr/bin/env python3
"""
X-ray Image Classifier using TorchXRayVision DenseNet121
Predicts pathologies from chest X-ray images.
Input: 224×224 pixels
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
from skimage.io import imread
import skimage

# Set local weights directory for portability
# This ensures all model weights are stored in the project directory
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "models_data"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Monkey patch torchxrayvision to use local weights directory
# This overrides the default ~/.torchxrayvision/models_data/ location
def _get_local_cache_dir():
    """Return the local project weights directory."""
    return str(WEIGHTS_DIR) + "/"

# Apply the patch before importing models
xrv.utils.get_cache_dir = _get_local_cache_dir


def load_model(weights="densenet121-res224-all", device=None):
    """
    Load pre-trained DenseNet121 model from torchxrayvision.
    
    Model weights are stored locally in the project's 'weights/' directory
    for portability. This allows the entire project to be copied as a single
    package with all dependencies.
    
    Args:
        weights: Model weights to use (default: densenet121-res224-all)
                 Available options:
                 - densenet121-res224-all: All datasets combined (recommended)
                 - densenet121-res224-rsna: RSNA Pneumonia Detection
                 - densenet121-res224-nih: NIH ChestX-ray14
                 - densenet121-res224-pc: PadChest
                 - densenet121-res224-chex: CheXpert
                 - densenet121-res224-mimic_nb: MIMIC-CXR (frontal view)
                 - densenet121-res224-mimic_ch: MIMIC-CXR (chest)
        device: Device to load model on (cuda/cpu)
    
    Returns:
        Loaded model and device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {weights}")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Device: {device}")
    print(f"Architecture: DenseNet121 (224×224 input)")
    
    # Load pre-trained DenseNet121 model
    # Weights will be downloaded to local weights/ directory if not present
    model = xrv.models.DenseNet(weights=weights)
    model = model.to(device)
    model.eval()
    
    return model, device


def preprocess_image(image_path):
    """
    Preprocess the X-ray image for DenseNet121 model input (224×224).
    Uses the same preprocessing pipeline as the torchxrayvision library.
    
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
    # This matches the preprocessing used during training
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
    Generate predictions for the input image.
    
    Args:
        model: Loaded model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
    
    Returns:
        Dictionary of pathology predictions
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Get pathology labels
        pathologies = model.pathologies
        
        # Create results dictionary
        results = {pathology: float(prob) for pathology, prob in zip(pathologies, probs)}
        
    return results


def print_predictions(results, threshold=0.5):
    """
    Print predictions in a formatted way.
    
    Args:
        results: Dictionary of pathology predictions
        threshold: Probability threshold for positive prediction
    """
    print("\n" + "="*60)
    print("X-RAY CLASSIFICATION RESULTS")
    print("="*60)
    
    # Sort by probability (highest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nAll Predictions (sorted by probability):")
    print("-"*60)
    for pathology, prob in sorted_results:
        status = "POSITIVE" if prob >= threshold else "negative"
        print(f"{pathology:30s}: {prob:.4f} ({status})")
    
    print("\n" + "="*60)
    positive_findings = [p for p, prob in results.items() if prob >= threshold]
    if positive_findings:
        print(f"\nPositive Findings (≥{threshold}):")
        for finding in positive_findings:
            print(f"  • {finding}: {results[finding]:.4f}")
    else:
        print(f"\nNo positive findings above threshold ({threshold})")
    print("="*60 + "\n")


def main():
    """Main function to run the classifier."""
    parser = argparse.ArgumentParser(
        description="Classify chest X-ray images using TorchXRayVision DenseNet121",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python src/classify_densenet121.py --input xray_image.jpg
  python src/classify_densenet121.py --input xray.png --threshold 0.6
  python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-rsna
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
        default='densenet121-res224-all',
        help='Model weights to use (default: densenet121-res224-all). '
             'Options: densenet121-res224-all, densenet121-res224-rsna, '
             'densenet121-res224-nih, densenet121-res224-pc, densenet121-res224-chex, '
             'densenet121-res224-mimic_nb, densenet121-res224-mimic_ch'
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
        model, device = load_model(weights=args.weights, device=device)
        
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

