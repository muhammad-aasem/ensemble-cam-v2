#!/usr/bin/env python3
"""
Generate Class Activation Maps (CAMs) for TorchXRayVision classifiers.
Standalone CLI tool that works with any TorchXRayVision model.
Generates P0 (overlay), P1 (heatmap), P2 (red channel), and P3 (binary mask).

Usage:
    python src/generate_cams.py \\
        --input data/INPUT/xray.jpg \\
        --model densenet121 \\
        --weights densenet121-res224-all \\
        --method gradcam++ \\
        --output OUTPUT/ \\
        --top-k 3
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchxrayvision as xrv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from cam_utils import (
    load_model_for_cam,
    preprocess_image_for_cam,
    generate_cam,
    generate_multi_class_cams,
    save_cam_outputs,
    save_comparison_figure,
    save_metadata,
    CAM_METHODS
)

# Set local weights directory
PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "models_data"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

def _get_local_cache_dir():
    """Return the local project weights directory."""
    return str(WEIGHTS_DIR) + "/"

xrv.utils.get_cache_dir = _get_local_cache_dir


def generate_red_channel(input_path, output_path=None):
    """
    Extract and save only the red channel from an input image.
    
    Args:
        input_path: Path to input heatmap image
        output_path: Path to save output (default: input_path with '_P2' suffix)
    
    Returns:
        Path to output image
    """
    # Read image
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    # Create output path if not provided
    if output_path is None:
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}_P2{input_path.suffix}"
    
    # Extract channels (OpenCV uses BGR format)
    blue, green, red = cv2.split(img)
    
    # Create image with only red channel (green and blue set to zero)
    red_only = cv2.merge([np.zeros_like(blue), np.zeros_like(green), red])
    
    # Save result
    cv2.imwrite(str(output_path), red_only)
    
    return output_path


def generate_binary_mask(input_path, output_path=None, threshold_value=127):
    """
    Generate binary mask from input image using thresholding.
    
    Args:
        input_path: Path to input heatmap image
        output_path: Path to save output (default: input_path with '_P3' suffix)
        threshold_value: Threshold value for binary conversion (default: 127)
    
    Returns:
        Path to output image
    """
    # Read image
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    # Create output path if not provided
    if output_path is None:
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}_P3{input_path.suffix}"
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold: cv2.threshold(gray, threshold_value, max_value, threshold_type)
    # threshold_type = 0 is cv2.THRESH_BINARY
    _, binary_mask = cv2.threshold(gray, threshold_value, 255, 0)
    
    # Save result
    cv2.imwrite(str(output_path), binary_mask)
    
    return output_path


def get_model_config(model_name: str) -> dict:
    """Get default configuration for a model."""
    configs = {
        'densenet121': {
            'type': 'densenet',
            'default_weights': 'densenet121-res224-all',
            'input_size': 224,
            'available_weights': [
                'densenet121-res224-all',
                'densenet121-res224-rsna',
                'densenet121-res224-nih',
                'densenet121-res224-pc',
                'densenet121-res224-chex',
                'densenet121-res224-mimic_nb',
                'densenet121-res224-mimic_ch'
            ]
        },
        'resnet50': {
            'type': 'resnet',
            'default_weights': 'resnet50-res512-all',
            'input_size': 512,
            'available_weights': ['resnet50-res512-all']
        },
        'jfhealthcare': {
            'type': 'jfhealthcare',
            'default_weights': None,  # No weights needed
            'input_size': 512,
            'available_weights': []
        },
        'mobilenetv3': {
            'type': 'mobilenetv3',
            'default_weights': 'weights/train_weight_mobilenet/mobilenetv3_best.pt',
            'input_size': 224,
            'available_weights': ['mobilenetv3_best.pt', 'mobilenetv3_last.pt']
        }
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(configs.keys())}")
    
    return configs[model_name]


def main():
    """Main function for CAM generation."""
    parser = argparse.ArgumentParser(
        description="Generate Class Activation Maps for X-ray classifiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate CAM for a specific pathology class (P0, P1, P2, P3 by default)
  python src/generate_cams.py --input xray.jpg --model densenet121 --target-class "Cardiomegaly"
  python src/generate_cams.py --input xray.jpg --model resnet50 --target-class "pneumonia"
  
  # Generate CAM for specific class with different method
  python src/generate_cams.py \\
      --input xray.jpg \\
      --model densenet121 \\
      --target-class "Fibrosis" \\
      --method scorecam
  
  # Generate top-K CAMs (if no target-class specified)
  python src/generate_cams.py --input xray.jpg --model densenet121 --top-k 5
  
  # Try different CAM methods
  python src/generate_cams.py --input xray.jpg --model resnet50 --method hirescam
  
  # Use smoothing for better quality (slower)
  python src/generate_cams.py --input xray.jpg --model densenet121 --smooth
  
  # Only generate CAMs for high-confidence predictions (when using top-k)
  python src/generate_cams.py --input xray.jpg --model densenet121 --threshold 0.6
  
  # Skip P2 and P3 generation (only P0 and P1)
  python src/generate_cams.py --input xray.jpg --model densenet121 --target-class "Cardiomegaly" --no-p2 --no-p3
  
  # Custom threshold for P3 binary mask
  python src/generate_cams.py --input xray.jpg --model densenet121 --target-class "Cardiomegaly" --p3-threshold 150

Output files:
  P0: {{model}}_{{pathology}}_P0.png (overlay - heatmap on original image)
  P1: {{model}}_{{pathology}}_P1.png (heatmap only)
  P2: {{model}}_{{pathology}}_P2.png (red channel from P1)
  P3: {{model}}_{{pathology}}_P3.png (binary mask from P1)

Available CAM methods: {}
Available models: densenet121, resnet50, jfhealthcare, mobilenetv3
        """.format(', '.join(CAM_METHODS.keys()))
    )
    
    # Required arguments
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input X-ray image'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['densenet121', 'resnet50', 'jfhealthcare', 'mobilenetv3'],
        help='Model to use for CAM generation'
    )
    
    # Optional arguments
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Model weights to use (uses model default if not specified)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='gradcam++',
        choices=list(CAM_METHODS.keys()),
        help='CAM method to use (default: gradcam++)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='OUTPUT',
        help='Output directory (default: OUTPUT/)'
    )
    
    parser.add_argument(
        '--target-class',
        type=str,
        default=None,
        help='Generate CAM for a specific pathology class (e.g., "Cardiomegaly", "Pneumonia"). '
             'Case-insensitive. If not specified, generates for top-3 predictions.'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Generate CAMs for top-K predictions (default: 3). Ignored if --target-class is specified.'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Only generate CAMs for predictions above this threshold (default: 0.0). '
             'Ignored if --target-class is specified.'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use (default: auto)'
    )
    
    parser.add_argument(
        '--smooth',
        action='store_true',
        help='Apply smoothing for better quality CAMs (slower)'
    )
    
    parser.add_argument(
        '--no-overlay',
        action='store_true',
        help='Skip generating overlay images'
    )
    
    parser.add_argument(
        '--no-heatmap',
        action='store_true',
        help='Skip generating heatmap-only images'
    )
    
    parser.add_argument(
        '--no-p2',
        action='store_true',
        help='Skip generating P2 (red channel) images'
    )
    
    parser.add_argument(
        '--no-p3',
        action='store_true',
        help='Skip generating P3 (binary mask) images'
    )
    
    parser.add_argument(
        '--p3-threshold',
        type=int,
        default=127,
        help='Threshold value for P3 binary mask (0-255, default: 127)'
    )
    
    parser.add_argument(
        '--comparison-figure',
        action='store_true',
        help='Generate a single figure comparing all CAMs'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input image not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Get model configuration
    model_config = get_model_config(args.model)
    
    # Determine weights to use
    if args.weights:
        weights = args.weights
    else:
        weights = model_config['default_weights']
        if weights is None:
            # JF Healthcare model - no weights needed
            print(f"Using {args.model} model (no weights specification needed)")
        else:
            print(f"Using default weights: {weights}")
    
    # Setup output directory
    output_root = Path(args.output)
    
    # Create subdirectory based on image filename
    image_name = input_path.stem  # Filename without extension
    output_dir = output_root / image_name
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Copy input image to output directory for reference
    import shutil
    input_image_dest = output_dir / "_input_image_.png"
    if not input_image_dest.exists():
        shutil.copy2(input_path, input_image_dest)
        print(f"Saved input image copy: {input_image_dest.name}")
    
    # Clean up existing files for this model
    if output_dir.exists():
        if args.target_class:
            # Delete only files for this model and target class
            safe_target = args.target_class.replace(' ', '_').replace('/', '_')
            pattern = f"{args.model}_{safe_target}_*.png"
            
            matching_files = list(output_dir.glob(pattern))
            if matching_files:
                print(f"Removing {len(matching_files)} existing file(s) for {args.model} + {args.target_class}:")
                for file in matching_files:
                    print(f"  - {file.name}")
                    file.unlink()
            else:
                print(f"No existing files found for {args.model} + {args.target_class}")
        else:
            # In top-k mode, delete all files for this model
            pattern = f"{args.model}_*.png"
            matching_files = list(output_dir.glob(pattern))
            if matching_files:
                print(f"Removing {len(matching_files)} existing file(s) for {args.model}:")
                for file in matching_files:
                    print(f"  - {file.name}")
                    file.unlink()
            else:
                print(f"No existing files found for {args.model}")
    
    try:
        # Load model
        print(f"\nLoading model: {args.model}")
        print(f"Method: {args.method}")
        print(f"Device: {args.device}")
        
        model, device = load_model_for_cam(
            model_type=model_config['type'],
            weights=weights if weights else '',  # Empty string for jfhealthcare
            device=args.device
        )
        
        print(f"Model loaded on: {device}")
        print(f"Model pathologies: {len(model.pathologies)}")
        
        # Preprocess image
        print(f"\nPreprocessing image: {input_path}")
        print(f"Target size: {model_config['input_size']}×{model_config['input_size']}")
        
        image_tensor, image_viz = preprocess_image_for_cam(
            input_path,
            target_size=model_config['input_size']
        )
        
        # Move tensor to device
        image_tensor = image_tensor.to(device)
        
        # Check if target-class is specified
        if args.target_class:
            print(f"\nGenerating {args.method.upper()} for target class: {args.target_class}")
            
            # Find matching pathology (case-insensitive)
            target_pathology = None
            for pathology in model.pathologies:
                if pathology.lower() == args.target_class.lower():
                    target_pathology = pathology
                    break
            
            if not target_pathology:
                print(f"\n❌ Error: Target class '{args.target_class}' not found in model pathologies.", file=sys.stderr)
                print(f"\nAvailable pathologies for {args.model}:")
                for i, pathology in enumerate(model.pathologies, 1):
                    print(f"  {i}. {pathology}")
                sys.exit(1)
            
            # Get prediction for target class
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()[0]
            
            target_idx = model.pathologies.index(target_pathology)
            target_prob = float(probs[target_idx])
            
            print(f"Target pathology: {target_pathology}")
            print(f"Prediction probability: {target_prob:.4f} ({target_prob*100:.2f}%)")
            
            # Generate CAM for target class
            use_cuda = (str(device) == 'cuda')
            cam_heatmap = generate_cam(
                model=model,
                image_tensor=image_tensor,
                target_class_idx=target_idx,
                model_type=model_config['type'],
                method=args.method,
                use_cuda=use_cuda
            )
            
            cams_dict = {target_pathology: (target_prob, cam_heatmap)}
            
        else:
            # Generate CAMs for top-K predictions
            print(f"\nGenerating {args.method.upper()} for top-{args.top_k} predictions...")
            print(f"Threshold: {args.threshold}")
            
            use_cuda = (str(device) == 'cuda')
            
            cams_dict = generate_multi_class_cams(
                model=model,
                image_tensor=image_tensor,
                model_type=model_config['type'],
                method=args.method,
                top_k=args.top_k,
                threshold=args.threshold,
                use_cuda=use_cuda
            )
            
            if not cams_dict:
                print(f"\nNo predictions above threshold {args.threshold}. No CAMs generated.")
                sys.exit(0)
            
            print(f"\nGenerated {len(cams_dict)} CAMs:")
            for pathology, (prob, _) in cams_dict.items():
                print(f"  • {pathology}: {prob:.4f} ({prob*100:.2f}%)")
        
        # Save individual CAMs
        print(f"\nSaving CAMs to {output_dir}...")
        
        for pathology, (prob, cam_heatmap) in cams_dict.items():
            # Save P0 (overlay) and P1 (heatmap)
            save_cam_outputs(
                original_image=image_viz,
                cam_heatmap=cam_heatmap,
                pathology=pathology,
                probability=prob,
                output_dir=output_dir,
                method=args.method,
                classifier_name=args.model,  # Add classifier prefix
                include_overlay=not args.no_overlay,
                include_heatmap_only=not args.no_heatmap
            )
            
            # Generate P2 and P3 from P1 (heatmap)
            if not args.no_heatmap:
                # Construct paths
                safe_pathology = pathology.replace(' ', '_').replace('/', '_')
                prefix = f"{args.model}_" if args.model else ""
                p1_path = output_dir / f"{prefix}{safe_pathology}_P1.png"
                p2_path = output_dir / f"{prefix}{safe_pathology}_P2.png"
                p3_path = output_dir / f"{prefix}{safe_pathology}_P3.png"
                
                if p1_path.exists():
                    # Generate P2 (red channel) from P1
                    if not args.no_p2:
                        try:
                            generate_red_channel(p1_path, output_path=p2_path)
                            print(f"  ✓ Generated P2 (red channel): {p2_path.name}")
                        except Exception as e:
                            print(f"  ⚠ Warning: Failed to generate P2 for {pathology}: {e}")
                    
                    # Generate P3 (binary mask) from P1
                    if not args.no_p3:
                        try:
                            generate_binary_mask(p1_path, output_path=p3_path, threshold_value=args.p3_threshold)
                            print(f"  ✓ Generated P3 (binary mask): {p3_path.name}")
                        except Exception as e:
                            print(f"  ⚠ Warning: Failed to generate P3 for {pathology}: {e}")
        
        # Generate comparison figure if requested
        if args.comparison_figure:
            comparison_path = output_dir / f"comparison_{args.method}.png"
            print(f"Generating comparison figure: {comparison_path}")
            save_comparison_figure(
                original_image=image_viz,
                cams_dict=cams_dict,
                output_path=comparison_path,
                method=args.method
            )
        
        # Save metadata
        all_predictions = {}
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
            for pathology, prob in zip(model.pathologies, probs):
                all_predictions[pathology] = float(prob)
        
        save_metadata(
            output_dir=output_dir,
            image_path=str(input_path),
            model_name=args.model,
            model_type=model_config['type'],
            weights=weights if weights else 'N/A',
            method=args.method,
            predictions=all_predictions,
            top_k_cams=list(cams_dict.keys())
        )
        
        print(f"\n✅ CAM generation complete!")
        print(f"   Output directory: {output_dir}")
        print(f"   Files generated:")
        
        for file in sorted(output_dir.iterdir()):
            print(f"     - {file.name}")
        
    except Exception as e:
        print(f"\n❌ Error during CAM generation: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

