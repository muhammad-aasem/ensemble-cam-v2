#!/usr/bin/env python3
"""
X-ray Image Classifier using TorchXRayVision JF Healthcare DenseNet Baseline Model
Predicts 5 key pathologies from chest X-ray images.
Input: 512×512 pixels (high resolution)
Specialized model from JF Healthcare
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


def load_model(device=None):
    """
    Load pre-trained JF Healthcare DenseNet baseline model from torchxrayvision.
    
    This is a specialized baseline model trained by JF Healthcare focusing on
    5 key pathologies. Model weights are stored locally in the project's 
    'weights/' directory for portability.
    
    Args:
        device: Device to load model on (cuda/cpu)
    
    Returns:
        Loaded model and device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: JF Healthcare DenseNet (Baseline)")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Device: {device}")
    print(f"Architecture: DenseNet121 (512×512 input)")
    print(f"Specialized for: 5 key pathologies")
    
    # Load pre-trained JF Healthcare baseline model
    # Weights will be downloaded to local weights/ directory if not present
    model = xrv.baseline_models.jfhealthcare.DenseNet()
    model = model.to(device)
    model.eval()
    
    return model, device


def preprocess_image(image_path):
    """
    Preprocess the X-ray image for JF Healthcare DenseNet input (512×512).
    Uses the same preprocessing pipeline as the torchxrayvision library.
    
    Args:
        image_path: Path to the X-ray image
    
    Returns:
        Preprocessed image tensor (1, 1, 512, 512)
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
    
    # Resize using skimage to 512×512 (native resolution for this model)
    from skimage.transform import resize
    img = resize(img, (512, 512), anti_aliasing=True, preserve_range=True)
    
    # Normalize pixel values using torchxrayvision's normalize function
    # This matches the preprocessing used during training
    img = xrv.datasets.normalize(img, maxval=255, reshape=False)
    
    # Ensure we have the right shape (512, 512)
    if img.shape != (512, 512):
        raise ValueError(f"Image shape is {img.shape}, expected (512, 512)")
    
    # Add batch and channel dimensions
    # Shape: (1, 1, 512, 512)
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
    print("JF Healthcare DenseNet Baseline Model")
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
    
    print("\n" + "="*60)
    print("Note: This model detects 5 key pathologies")
    print("For comprehensive 18-pathology detection, use DenseNet121 or ResNet50")
    print("="*60 + "\n")


def main():
    """Main function to run the classifier."""
    parser = argparse.ArgumentParser(
        description="Classify chest X-rays using JF Healthcare DenseNet (5 pathologies)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python src/classify_jfhealthcare.py --input xray_image.jpg
  python src/classify_jfhealthcare.py --input xray.png --threshold 0.6
  
About this model:
  - Baseline model from JF Healthcare
  - Specialized for 5 key pathologies: Cardiomegaly, Edema, Consolidation, 
    Atelectasis, Effusion
  - Uses 512×512 high-resolution input
  - DenseNet121 architecture
  
For comprehensive detection (18 pathologies):
  - Use classify_densenet121.py (224×224, faster)
  - Use classify_resnet50.py (512×512, high-res)
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
        model, device = load_model(device=device)
        
        # Preprocess image
        print(f"Processing image: {args.input}")
        print(f"Input size: 512×512 (native resolution)")
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

