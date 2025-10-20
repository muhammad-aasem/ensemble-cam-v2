#!/usr/bin/env python3
"""
Shared CAM (Class Activation Map) utilities for TorchXRayVision models.
Used by both CLI (generate_cams.py) and Web UI (app.py).

Supports multiple CAM methods: GradCAM, GradCAM++, ScoreCAM, HiResCAM, etc.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torchxrayvision as xrv
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    HiResCAM,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    LayerCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Map method names to classes
CAM_METHODS = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'scorecam': ScoreCAM,
    'hirescam': HiResCAM,
    'ablationcam': AblationCAM,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'layercam': LayerCAM,
}


def get_target_layer(model, model_type: str):
    """
    Get the appropriate target layer for CAM generation based on model type.
    
    Args:
        model: Loaded TorchXRayVision model
        model_type: Type of model ('densenet', 'resnet', 'jfhealthcare', 'mobilenetv3')
    
    Returns:
        List containing the target layer(s) for CAM
    """
    if model_type == 'densenet':
        # DenseNet: Use last layer of features
        return [model.features[-1]]
    elif model_type == 'jfhealthcare':
        # JF Healthcare baseline: Features are nested under model.module.backbone
        return [model.model.module.backbone.features[-1]]
    elif model_type == 'resnet':
        # ResNet: TorchXRayVision ResNet has layers under 'model' attribute
        return [model.model.layer4[-1]]
    elif model_type == 'mobilenetv3':
        # MobileNetV3: Use last layer of features
        return [model.features[-1]]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model_for_cam(model_type: str, weights: str, device: str = 'auto'):
    """
    Load a TorchXRayVision model for CAM generation.
    
    Args:
        model_type: Type of model ('densenet', 'resnet', 'jfhealthcare', 'mobilenetv3')
        weights: Model weights to use (path for mobilenetv3)
        device: Device to load model on ('cuda', 'cpu', 'auto')
    
    Returns:
        Tuple of (model, device)
    """
    # Set device
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load model based on type
    if model_type == 'densenet':
        model = xrv.models.DenseNet(weights=weights)
    elif model_type == 'resnet':
        model = xrv.models.ResNet(weights=weights)
    elif model_type == 'jfhealthcare':
        model = xrv.baseline_models.jfhealthcare.DenseNet()
    elif model_type == 'mobilenetv3':
        # Load custom MobileNetV3 model
        from pathlib import Path
        import sys
        import torch.nn as nn
        
        # Add src to path if not already there
        src_path = Path(__file__).parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        # Import MobileNetV3XRay from classify_mobilenetv3
        from classify_mobilenetv3 import MobileNetV3XRay, TRAIN_DIR
        
        # Determine weights path
        if weights and Path(weights).exists():
            weights_path = Path(weights)
        else:
            weights_path = TRAIN_DIR / "mobilenetv3_best.pt"
        
        if not weights_path.exists():
            raise FileNotFoundError(
                f"MobileNetV3 weights not found at {weights_path}. "
                f"Please train the model first using: "
                f"uv run python src/train_mobilenet.py --epochs 50 --imgpath /path/to/images --csvpath /path/to/csv"
            )
        
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')
        num_classes = len(checkpoint['pathologies'])
        
        # Create and load model
        model = MobileNetV3XRay(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.pathologies = checkpoint['pathologies']
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    return model, device


def generate_cam(
    model,
    image_tensor: torch.Tensor,
    target_class_idx: int,
    model_type: str,
    method: str = 'gradcam++',
    use_cuda: bool = True,
    aug_smooth: bool = False,
    eigen_smooth: bool = False
) -> np.ndarray:
    """
    Generate Class Activation Map for a specific pathology.
    
    Args:
        model: Loaded TorchXRayVision model
        image_tensor: Preprocessed image tensor (1, 1, H, W)
        target_class_idx: Index of target pathology class
        model_type: Type of model ('densenet', 'resnet', 'jfhealthcare')
        method: CAM method to use (default: 'gradcam++')
        use_cuda: Whether to use GPU if available
        aug_smooth: Apply test-time augmentation for smoother CAMs
        eigen_smooth: Apply PCA smoothing for noise reduction
    
    Returns:
        CAM heatmap as numpy array (H, W) with values in [0, 1]
    """
    # Get target layer
    target_layers = get_target_layer(model, model_type)
    
    # Get CAM class
    if method.lower() not in CAM_METHODS:
        raise ValueError(f"Unknown CAM method: {method}. Available: {list(CAM_METHODS.keys())}")
    
    CAMClass = CAM_METHODS[method.lower()]
    
    # Initialize CAM (newer versions don't use use_cuda parameter)
    cam = CAMClass(
        model=model,
        target_layers=target_layers
    )
    
    # Define target
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    # Generate CAM
    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets,
        aug_smooth=aug_smooth,
        eigen_smooth=eigen_smooth
    )
    
    return grayscale_cam[0]  # Return first (and only) image's CAM


def generate_multi_class_cams(
    model,
    image_tensor: torch.Tensor,
    model_type: str,
    method: str = 'gradcam++',
    top_k: int = 3,
    threshold: float = 0.0,
    use_cuda: bool = True
) -> Dict[str, Tuple[float, np.ndarray]]:
    """
    Generate CAMs for top-K predicted pathologies.
    
    Args:
        model: Loaded TorchXRayVision model
        image_tensor: Preprocessed image tensor (1, 1, H, W)
        model_type: Type of model
        method: CAM method to use
        top_k: Number of top predictions to generate CAMs for
        threshold: Only generate CAMs for predictions above this threshold
        use_cuda: Whether to use GPU
    
    Returns:
        Dictionary mapping pathology names to (probability, CAM) tuples
    """
    # Get predictions
    with torch.no_grad():
        image_tensor = image_tensor.to(model.device if hasattr(model, 'device') else 'cpu')
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Get pathology labels
    pathologies = model.pathologies
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    
    # Generate CAMs for top-K predictions above threshold
    cams = {}
    count = 0
    
    for idx in sorted_indices:
        if count >= top_k:
            break
        
        prob = float(probs[idx])
        if prob < threshold:
            continue
        
        pathology = pathologies[idx]
        
        # Generate CAM
        cam_heatmap = generate_cam(
            model=model,
            image_tensor=image_tensor,
            target_class_idx=idx,
            model_type=model_type,
            method=method,
            use_cuda=use_cuda
        )
        
        cams[pathology] = (prob, cam_heatmap)
        count += 1
    
    return cams


def create_overlay_image(
    original_image: np.ndarray,
    cam_heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Create an overlay of CAM heatmap on original image.
    
    Args:
        original_image: Original grayscale image (H, W) normalized to [0, 1]
        cam_heatmap: CAM heatmap (H, W) with values in [0, 1]
        alpha: Transparency of overlay (0 = transparent, 1 = opaque)
        colormap: OpenCV colormap to use for heatmap
    
    Returns:
        RGB image with CAM overlay (H, W, 3) with values in [0, 255]
    """
    # Ensure original image is 2D
    if len(original_image.shape) == 3:
        original_image = original_image.mean(axis=2)
    
    # Normalize original image to [0, 1]
    if original_image.max() > 1.0:
        original_image = original_image / 255.0
    
    # Convert grayscale to RGB
    rgb_image = np.stack([original_image] * 3, axis=-1)
    
    # Use pytorch_grad_cam's utility function
    cam_overlay = show_cam_on_image(
        rgb_image.astype(np.float32),
        cam_heatmap,
        use_rgb=True
    )
    
    return cam_overlay


def save_cam_outputs(
    original_image: np.ndarray,
    cam_heatmap: np.ndarray,
    pathology: str,
    probability: float,
    output_dir: Path,
    method: str,
    classifier_name: str = None,
    include_overlay: bool = True,
    include_heatmap_only: bool = True
):
    """
    Save CAM visualizations to disk.
    
    Args:
        original_image: Original image (H, W) normalized to [0, 1]
        cam_heatmap: CAM heatmap (H, W)
        pathology: Name of the pathology
        probability: Prediction probability
        output_dir: Directory to save outputs
        method: CAM method used
        classifier_name: Name of classifier (e.g., 'densenet121', 'resnet50')
        include_overlay: Whether to save overlay image
        include_heatmap_only: Whether to save heatmap only
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Safe filename from pathology name
    safe_pathology = pathology.replace(' ', '_').replace('/', '_')
    
    # Add classifier prefix if provided
    prefix = f"{classifier_name}_" if classifier_name else ""
    
    # Save overlay (P0) (clean - no annotations)
    if include_overlay:
        overlay_path = output_dir / f"{prefix}{safe_pathology}_P0.png"
        
        overlay_image = create_overlay_image(original_image, cam_heatmap)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay_image)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    # Save heatmap only (P1) (clean - no annotations)
    if include_heatmap_only:
        heatmap_path = output_dir / f"{prefix}{safe_pathology}_P1.png"
        
        plt.figure(figsize=(8, 8))
        plt.imshow(cam_heatmap, cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()


def save_comparison_figure(
    original_image: np.ndarray,
    cams_dict: Dict[str, Tuple[float, np.ndarray]],
    output_path: Path,
    method: str
):
    """
    Save a comparison figure showing original image and multiple CAMs.
    
    Args:
        original_image: Original X-ray image (H, W)
        cams_dict: Dictionary mapping pathology -> (probability, CAM)
        output_path: Path to save the comparison figure
        method: CAM method used
    """
    n_cams = len(cams_dict)
    
    if n_cams == 0:
        return
    
    # Create subplots: original + N CAMs
    fig, axes = plt.subplots(1, n_cams + 1, figsize=(5 * (n_cams + 1), 5))
    
    if n_cams == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
    
    # Show original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original X-ray')
    axes[0].axis('off')
    
    # Show each CAM
    for idx, (pathology, (prob, cam_heatmap)) in enumerate(cams_dict.items()):
        overlay = create_overlay_image(original_image, cam_heatmap)
        axes[idx + 1].imshow(overlay)
        axes[idx + 1].set_title(f'{pathology}\n{prob:.2%}')
        axes[idx + 1].axis('off')
    
    plt.suptitle(f'Class Activation Maps ({method.upper()})', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metadata(
    output_dir: Path,
    image_path: str,
    model_name: str,
    model_type: str,
    weights: str,
    method: str,
    predictions: Dict[str, float],
    top_k_cams: List[str]
):
    """
    Save or update metadata about the CAM generation process.
    Merges with existing metadata if present, allowing multiple models' data to coexist.
    
    Args:
        output_dir: Output directory
        image_path: Path to input image
        model_name: Name of model (e.g., 'densenet121', 'resnet50')
        model_type: Type of model used
        weights: Model weights used
        method: CAM method used
        predictions: All predictions (pathology -> probability)
        top_k_cams: List of pathologies CAMs were generated for
    """
    metadata_path = output_dir / 'metadata.json'
    
    # Load existing metadata if it exists
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            'input_image': str(image_path),
            'cams': {}
        }
    
    # Check if this model already has an entry
    if model_name in metadata['cams']:
        # Merge cams_generated_for lists (keep unique entries)
        existing_cams = set(metadata['cams'][model_name].get('cams_generated_for', []))
        new_cams = set(top_k_cams)
        merged_cams = sorted(list(existing_cams.union(new_cams)))
    else:
        merged_cams = top_k_cams
    
    # Update or add entry for this model
    metadata['cams'][model_name] = {
        'model_type': model_type,
        'weights': weights,
        'cam_method': method,
        'predictions': predictions,
        'cams_generated_for': merged_cams,
    }
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def preprocess_image_for_cam(image_path: Path, target_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess X-ray image for CAM generation.
    Returns both the tensor for model input and normalized array for visualization.
    
    Args:
        image_path: Path to image
        target_size: Target size (224 or 512)
    
    Returns:
        Tuple of (image_tensor, normalized_image_array)
    """
    from skimage.io import imread
    from skimage.transform import resize
    
    # Load image
    img = imread(image_path)
    
    # Convert to grayscale if RGB/RGBA
    if len(img.shape) > 2:
        if img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        img = img.mean(axis=2)
    
    # Convert to float32
    img = img.astype(np.float32)
    
    # Normalize to [0, 255] range if needed
    if img.max() <= 1.0:
        img = img * 255.0
    
    # Resize to target size
    img_resized = resize(img, (target_size, target_size), 
                        anti_aliasing=True, preserve_range=True)
    
    # Normalize for model input
    img_normalized = xrv.datasets.normalize(img_resized, maxval=255, reshape=False)
    
    # Create tensor for model
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)
    
    # Create normalized array for visualization (0 to 1)
    img_viz = img_resized / 255.0
    
    return img_tensor, img_viz

