#!/usr/bin/env python3
"""
Ensemble-CAM: Multi-Model X-ray Classifier Web UI
Streamlit-based interface for running multiple X-ray classifiers on chest X-ray images
"""

import os
import sys
from pathlib import Path
import tomli  # For TOML parsing (Python 3.11+ uses tomllib)
import numpy as np
import torch
import torchxrayvision as xrv
from skimage.io import imread
from skimage.transform import resize
import streamlit as st
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
from cam_utils import (
    preprocess_image_for_cam,
    generate_cam,
    create_overlay_image
)

# Set up project paths
PROJECT_ROOT = Path(__file__).parent
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "models_data"
INPUT_DIR = PROJECT_ROOT / "data" / "INPUT"
OUTPUT_DIR = PROJECT_ROOT / "OUTPUT"
CONFIG_FILE = PROJECT_ROOT / "config_classifiers.toml"

# Monkey patch torchxrayvision to use local weights directory
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
def _get_local_cache_dir():
    """Return the local project weights directory."""
    return str(WEIGHTS_DIR) + "/"
xrv.utils.get_cache_dir = _get_local_cache_dir


# ============================================================================
# Configuration Loading
# ============================================================================

@st.cache_data
def load_config():
    """Load classifier configuration from TOML file."""
    with open(CONFIG_FILE, 'rb') as f:
        config = tomli.load(f)
    return config


# ============================================================================
# Model Management
# ============================================================================

@st.cache_resource
def load_classifier(model_type, weights=None):
    """
    Load a classifier model based on type and weights.
    Cached to avoid reloading on every interaction.
    
    Args:
        model_type: Type of model (densenet, resnet, jfhealthcare)
        weights: Specific weights to load (None for jfhealthcare)
    
    Returns:
        Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "densenet":
        model = xrv.models.DenseNet(weights=weights)
    elif model_type == "resnet":
        model = xrv.models.ResNet(weights=weights)
    elif model_type == "jfhealthcare":
        model = xrv.baseline_models.jfhealthcare.DenseNet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    return model, device


# ============================================================================
# CAM Generation
# ============================================================================

def get_cam_paths(image_path, clf_key, target_class):
    """
    Get paths for CAM images (heatmap and overlay).
    
    Args:
        image_path: Path to input image
        clf_key: Classifier key (e.g., 'densenet121')
        target_class: Target pathology class
    
    Returns:
        Tuple of (heatmap_path, overlay_path, output_dir)
    """
    # Create output directory based on image name
    image_stem = Path(image_path).stem
    output_dir = OUTPUT_DIR / image_stem
    
    # CAM image paths with new P0/P1 naming convention
    # P0 = overlay, P1 = heatmap
    overlay_path = output_dir / f"{clf_key}_{target_class}_P0.png"
    heatmap_path = output_dir / f"{clf_key}_{target_class}_P1.png"
    
    return heatmap_path, overlay_path, output_dir


def cams_exist(image_path, clf_key, target_class):
    """
    Check if CAM images already exist for given parameters.
    
    Args:
        image_path: Path to input image
        clf_key: Classifier key
        target_class: Target pathology class
    
    Returns:
        True if both heatmap and overlay exist
    """
    heatmap_path, overlay_path, _ = get_cam_paths(image_path, clf_key, target_class)
    return heatmap_path.exists() and overlay_path.exists()


@st.cache_data(show_spinner=False)
def generate_cam_for_class(_model, image_path, clf_key, model_type, input_size, target_class):
    """
    Generate CAM visualization for a specific class.
    Uses Streamlit caching to avoid regeneration.
    
    Args:
        _model: The classifier model (prefixed with _ to exclude from hash)
        image_path: Path to input image
        clf_key: Classifier key
        model_type: Model type (densenet, resnet, jfhealthcare)
        input_size: Input size for the model
        target_class: Target pathology class name
    
    Returns:
        Tuple of (heatmap_path, overlay_path) or (None, None) if error
    """
    try:
        # Check if CAMs already exist
        heatmap_path, overlay_path, output_dir = get_cam_paths(image_path, clf_key, target_class)
        
        if heatmap_path.exists() and overlay_path.exists():
            return str(heatmap_path), str(overlay_path)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get target class index from pathology name
        try:
            target_class_idx = list(_model.pathologies).index(target_class)
        except ValueError:
            st.error(f"Pathology '{target_class}' not found in model pathologies")
            return None, None
        
        # Preprocess image
        image_tensor, original_image = preprocess_image_for_cam(Path(image_path), input_size)
        
        # Check for CUDA availability
        use_cuda = torch.cuda.is_available()
        
        # Generate CAM
        cam = generate_cam(
            model=_model,
            image_tensor=image_tensor,
            target_class_idx=target_class_idx,
            model_type=model_type,
            method='gradcam++',
            use_cuda=use_cuda
        )
        
        # Create overlay
        overlay_image = create_overlay_image(original_image, cam)
        
        # Save images
        from matplotlib import pyplot as plt
        
        # Save heatmap
        plt.figure(figsize=(8, 8))
        plt.imshow(cam, cmap='jet')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        # Save overlay
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay_image)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        return str(heatmap_path), str(overlay_path)
        
    except Exception as e:
        st.error(f"CAM generation failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


# ============================================================================
# Image Processing
# ============================================================================

def preprocess_image(image_path, target_size):
    """
    Preprocess X-ray image for model input.
    
    Args:
        image_path: Path to image
        target_size: Target size (224 or 512)
    
    Returns:
        Preprocessed image tensor (1, 1, H, W)
    """
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
    img = resize(img, (target_size, target_size), anti_aliasing=True, preserve_range=True)
    
    # Normalize using torchxrayvision's method
    img = xrv.datasets.normalize(img, maxval=255, reshape=False)
    
    # Add batch and channel dimensions
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    
    return img


def predict_with_model(model, image_tensor, device):
    """
    Run prediction with a model.
    
    Args:
        model: Loaded model
        image_tensor: Preprocessed image
        device: Device to run on
    
    Returns:
        Dictionary of pathology predictions
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        pathologies = model.pathologies
        results = {pathology: float(prob) for pathology, prob in zip(pathologies, probs)}
    
    return results


# ============================================================================
# UI Components
# ============================================================================

def get_available_images():
    """Get list of available images from INPUT directory."""
    if not INPUT_DIR.exists():
        return []
    
    # Supported image formats
    formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    images = []
    for fmt in formats:
        images.extend(INPUT_DIR.glob(fmt))
    
    return sorted([img.name for img in images])


def display_image_selector(config):
    """Display image selection panel with evaluation controls."""
    st.header("📁 Input")
    
    images = get_available_images()
    
    if not images:
        st.warning(f"No images found in `{INPUT_DIR.relative_to(PROJECT_ROOT)}`")
        st.info("Please add X-ray images to the INPUT directory.")
        return None
    
    # Image selector
    selected_image = st.selectbox(
        "Select an X-ray image:",
        images,
        key="image_selector"
    )
    
    if selected_image:
        image_path = INPUT_DIR / selected_image
        
        # Display image
        img = Image.open(image_path)
        st.image(img, caption=selected_image, use_container_width=True)
        
        # Image info
        st.caption(f"📏 Size: {img.size[0]}×{img.size[1]} pixels")
        
        st.divider()
        
        # Global evaluation controls (moved from Column 2)
        st.caption("🎯 Global Evaluation")
        
        # Get all unique pathologies from config
        all_pathologies = set()
        for clf_config in config['classifiers'].values():
            # Standard pathologies for most models (18 classes)
            standard_pathologies = [
                "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
                "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
                "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass",
                "Hernia", "Lung Lesion", "Fracture", "Lung Opacity",
                "Enlarged Cardiomediastinum"
            ]
            all_pathologies.update(standard_pathologies)
        
        # Dropdown for pathology selection
        eval_class = st.selectbox(
            "Select pathology class:",
            sorted(list(all_pathologies)),
            key="global_eval_class_selector",
            label_visibility="collapsed"
        )
        
        # Eval button
        if st.button("🔍 Evaluate All Classifiers", key="global_eval_btn", use_container_width=True, type="primary"):
            # Store the global evaluation class
            st.session_state['global_eval_active'] = True
            st.session_state['global_eval_target_class'] = eval_class
            st.success(f"Evaluating '{eval_class}' across all classifiers...")
        
        return image_path
    
    return None


def display_classifier_card(clf_key, clf_config, selected_image, global_eval_class=None):
    """
    Display a single classifier card with inline classification.
    
    Args:
        clf_key: Classifier key (e.g., 'densenet121')
        clf_config: Classifier configuration
        selected_image: Path to selected image
        global_eval_class: If set, auto-evaluate this class across all classifiers
    
    Returns:
        True if classifier was run successfully
    """
    with st.container(border=True):
        # Compact header: Model name (left) | Weight + Button (right)
        col_name, col_controls = st.columns([1.5, 2])
        
        with col_name:
            st.markdown(f"**{clf_config['name']}**")
            st.caption(f"{clf_config['num_pathologies']} classes")
        
        with col_controls:
            # Weight selection (if available) + Classify button on same row
            if clf_config['weights']:
                col_weight, col_btn = st.columns([2, 1])
                
                with col_weight:
                    selected_weight = st.selectbox(
                        "Weights",
                        clf_config['weights'],
                        index=0,
                        key=f"weight_{clf_key}",
                        label_visibility="collapsed"
                    )
                
                with col_btn:
                    # Classify button for this specific classifier (hide in global eval mode)
                    if not global_eval_class:
                        classify_btn = st.button(
                            "🔍",
                            key=f"classify_{clf_key}",
                            use_container_width=True,
                            type="primary",
                            help="Classify with this model"
                        )
                    else:
                        classify_btn = False
            else:
                # No weights, just button
                selected_weight = None
                if not global_eval_class:
                    classify_btn = st.button(
                        "🔍 Classify",
                        key=f"classify_{clf_key}",
                        use_container_width=True,
                        type="primary"
                    )
                else:
                    classify_btn = False
        
        # Global eval mode indicator (compact)
        if global_eval_class:
            st.caption(f"🎯 Evaluating: **{global_eval_class}**")
        
        # Initialize session state for this classifier
        result_key = f"result_{clf_key}"
        if result_key not in st.session_state:
            st.session_state[result_key] = None
        
        # Auto-classify in global eval mode if not already done
        if global_eval_class and st.session_state[result_key] is None and selected_image:
            with st.spinner(f"Running {clf_config['name']}..."):
                try:
                    # Load model
                    model, device = load_classifier(
                        clf_config['model_type'],
                        selected_weight
                    )
                    
                    # Preprocess image
                    img_tensor = preprocess_image(
                        selected_image,
                        clf_config['input_size']
                    )
                    
                    # Predict
                    results = predict_with_model(model, img_tensor, device)
                    
                    # Store results in session state
                    st.session_state[result_key] = results
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state[result_key] = None
        
        # Run classification when button is clicked (manual mode)
        if classify_btn:
            if not selected_image:
                st.error("⚠️ Please select an image first!")
            else:
                with st.spinner(f"Running {clf_config['name']}..."):
                    try:
                        # Load model
                        model, device = load_classifier(
                            clf_config['model_type'],
                            selected_weight
                        )
                        
                        # Preprocess image
                        img_tensor = preprocess_image(
                            selected_image,
                            clf_config['input_size']
                        )
                        
                        # Predict
                        results = predict_with_model(model, img_tensor, device)
                        
                        # Store results in session state
                        st.session_state[result_key] = results
                        
                        st.success("✅ Classification complete!")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.session_state[result_key] = None
        
        # Display inference results if available (inline roll-down)
        if st.session_state[result_key] is not None:
            st.divider()
            st.markdown("### 📊 Inference")
            
            results = st.session_state[result_key]
            
            # Initialize session state for selected class
            selected_class_key = f"selected_class_{clf_key}"
            if selected_class_key not in st.session_state:
                st.session_state[selected_class_key] = None
            
            # In global eval mode, show only the selected class
            if global_eval_class:
                # Auto-select the global eval class
                st.session_state[selected_class_key] = global_eval_class
                
                # Get confidence for the eval class
                if global_eval_class in results:
                    eval_confidence = results[global_eval_class] * 100
                    display_results = [(global_eval_class, results[global_eval_class])]
                else:
                    st.warning(f"⚠️ '{global_eval_class}' not available in this model's pathologies")
                    display_results = []
            else:
                # Normal mode: show top 3
                display_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Create 20%/80% split layout
            col_table, col_image = st.columns([1, 4])  # 20%/80% split
            
            # Left column (20%): Clickable predictions
            with col_table:
                if global_eval_class:
                    st.markdown(f"**{global_eval_class}:**")
                    # Show only the eval class with confidence
                    if display_results:
                        pathology, prob = display_results[0]
                        confidence = prob * 100
                        st.metric(
                            label="Confidence",
                            value=f"{confidence:.2f}%",
                            delta=None
                        )
                else:
                    st.markdown("**Top 3 Predictions:**")
                    st.caption("Click pathology to generate CAM")
                    
                    # Create clickable buttons for each prediction
                    for i, (pathology, prob) in enumerate(display_results, 1):
                        confidence = prob * 100
                        
                        # Button for each pathology (clickable)
                        button_type = "primary" if i == 1 else "secondary"
                        cam_exists = cams_exist(selected_image, clf_key, pathology) if selected_image else False
                        
                        # Button label with checkmark if CAM exists
                        label = f"{'✓ ' if cam_exists else ''}#{i} {pathology}\n({confidence:.2f}%)"
                        
                        if st.button(
                            label,
                            key=f"cam_btn_{clf_key}_{pathology}",
                            use_container_width=True,
                            type=button_type if i == 1 and st.session_state[selected_class_key] == pathology else "secondary"
                        ):
                            # Store selected class
                            st.session_state[selected_class_key] = pathology
            
            # Right column (80%): CAM visualization
            with col_image:
                st.markdown("**Visualization:**")
                
                # Checkboxes in same row
                col_chk1, col_chk2 = st.columns(2)
                with col_chk1:
                    show_overlay = st.checkbox(
                        "Show overlay",
                        value=True,
                        key=f"overlay_toggle_{clf_key}"
                    )
                with col_chk2:
                    draw_polygon_enabled = st.checkbox(
                        "Draw polygon",
                        value=False,
                        key=f"polygon_toggle_{clf_key}"
                    )
                
                # If a class is selected, generate and display CAM
                if st.session_state[selected_class_key] and selected_image:
                    target_class = st.session_state[selected_class_key]
                    
                    # Check if CAM already exists
                    if cams_exist(selected_image, clf_key, target_class):
                        st.info(f"ℹ️ Loading existing CAM for **{target_class}**")
                    else:
                        st.info(f"🔄 Generating CAM for **{target_class}**...")
                    
                    # Load model for CAM generation
                    model, device = load_classifier(
                        clf_config['model_type'],
                        selected_weight
                    )
                    
                    # Generate or load CAM
                    with st.spinner("Generating CAM..."):
                        heatmap_path, overlay_path = generate_cam_for_class(
                            _model=model,
                            image_path=str(selected_image),
                            clf_key=clf_key,
                            model_type=clf_config['model_type'],
                            input_size=clf_config['input_size'],
                            target_class=target_class
                        )
                    
                    # Display CAM image
                    if heatmap_path and overlay_path:
                        # Select base image
                        if show_overlay:
                            base_image_path = overlay_path
                            caption = f"GradCAM++ Overlay: {target_class}"
                        else:
                            base_image_path = heatmap_path
                            caption = f"GradCAM++ Heatmap: {target_class}"
                        
                        # Apply polygon drawing if enabled
                        if draw_polygon_enabled:
                            from src.draw import draw_polygon
                            from pathlib import Path
                            
                            # Create drawn version path
                            base_path = Path(base_image_path)
                            drawn_path = base_path.parent / f"{base_path.stem}_drawn{base_path.suffix}"
                            
                            # Generate drawn version if it doesn't exist or force regenerate
                            if not drawn_path.exists():
                                with st.spinner("Drawing polygon..."):
                                    draw_polygon(base_image_path, drawn_path, threshold=200, min_area=1000)
                            
                            # Display drawn version
                            cam_image = Image.open(drawn_path)
                            st.image(cam_image, caption=f"{caption} (Polygon)", use_container_width=True)
                        else:
                            # Display original
                            cam_image = Image.open(base_image_path)
                            st.image(cam_image, caption=caption, use_container_width=True)
                        
                        st.caption(
                            f"{'📊 Overlay mode: Heatmap overlaid on X-ray' if show_overlay else '🔥 Heatmap mode: Raw activation map'}"
                        )
                    else:
                        st.error("Failed to generate CAM visualization")
                
                else:
                    # Placeholder when no class is selected
                    st.info("👈 Click a pathology label to generate CAM visualization")
                    
                    # Add placeholder box
                    st.markdown(
                        """
                        <div style='border: 2px dashed #ccc; padding: 20px; text-align: center; 
                                    background-color: #f9f9f9; border-radius: 5px; min-height: 200px;
                                    display: flex; align-items: center; justify-content: center;'>
                            <div>
                                <p style='color: #666; margin: 0;'>📊 Heatmap visualization</p>
                                <p style='color: #999; font-size: 0.9em; margin: 5px 0 0 0;'>
                                    Click a pathology to see GradCAM++ activation maps
                                </p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


def display_classifiers_panel(config, selected_image):
    """Display all classifier cards."""
    
    # Simple header
    st.subheader("🤖 Classifiers")
    st.divider()
    
    classifiers_config = config['classifiers']
    
    # Check if global eval mode is active
    global_eval_active = st.session_state.get('global_eval_active', False)
    global_eval_class = st.session_state.get('global_eval_target_class', None)
    
    # Display classifier cards
    for clf_key, clf_config in classifiers_config.items():
        display_classifier_card(
            clf_key, 
            clf_config, 
            selected_image,
            global_eval_class=global_eval_class if global_eval_active else None
        )


def create_ensemble_cam(config, selected_image, target_class, transparency_weights=None):
    """
    Create ensemble CAM by overlaying heatmaps from all classifiers.
    
    Args:
        config: Configuration dict with classifiers
        selected_image: Path to selected image
        target_class: Target pathology class
        transparency_weights: Dict of {classifier_name: weight (0-100)}
    
    Returns:
        Tuple of (PIL Image heatmap, PIL Image overlay, list of classifier names, dict of confidences)
    """
    import cv2
    
    # Collect all CAM heatmap paths and confidences
    cam_paths = []
    classifier_names = []
    confidences = {}
    
    for clf_key in config['classifiers'].keys():
        heatmap_path, _, _ = get_cam_paths(selected_image, clf_key, target_class)
        if heatmap_path.exists():
            cam_paths.append(heatmap_path)
            classifier_names.append(clf_key)
            
            # Get confidence from session state
            result_key = f"result_{clf_key}"
            if result_key in st.session_state and st.session_state[result_key]:
                results = st.session_state[result_key]
                if target_class in results:
                    confidences[clf_key] = results[target_class] * 100
                else:
                    confidences[clf_key] = 0.0
            else:
                confidences[clf_key] = 0.0
    
    if len(cam_paths) == 0:
        return None, None, [], {}
    
    # Load all CAM images
    cams = []
    for path in cam_paths:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cams.append(img)
    
    # Ensure all images are same size
    target_size = cams[0].shape[:2]
    for i in range(len(cams)):
        if cams[i].shape[:2] != target_size:
            cams[i] = cv2.resize(cams[i], (target_size[1], target_size[0]))
    
    # Apply transparency weights if provided
    if transparency_weights is None:
        # Equal weights (50% each)
        weights = [0.5] * len(cams)
    else:
        # Normalize weights to sum to 1.0
        weights = []
        total_weight = 0
        for clf_name in classifier_names:
            weight = transparency_weights.get(clf_name, 50) / 100.0  # Convert 0-100 to 0-1
            weights.append(weight)
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(cams)] * len(cams)
    
    # Create ensemble by blending all CAMs with weighted transparency
    ensemble = np.zeros_like(cams[0], dtype=np.float32)
    
    for cam, weight in zip(cams, weights):
        ensemble += cam.astype(np.float32) * weight
    
    ensemble = np.clip(ensemble, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image (heatmap only)
    from PIL import Image as PILImage
    ensemble_heatmap = PILImage.fromarray(ensemble)
    
    # Create overlay version (heatmap on original image)
    # Load original image
    original_img = cv2.imread(str(selected_image))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Resize original to match ensemble size
    if original_img.shape[:2] != ensemble.shape[:2]:
        original_img = cv2.resize(original_img, (ensemble.shape[1], ensemble.shape[0]))
    
    # Blend: 60% original + 40% heatmap
    overlay = cv2.addWeighted(original_img, 0.6, ensemble, 0.4, 0)
    ensemble_overlay = PILImage.fromarray(overlay)
    
    return ensemble_heatmap, ensemble_overlay, classifier_names, confidences


def display_output_placeholder(config, selected_image):
    """Display ensemble CAM visualization."""
    st.header("🔥 Ensemble CAM")
    
    # Check if global eval mode is active
    global_eval_class = st.session_state.get('global_eval_target_class', None)
    
    if global_eval_class and selected_image:
        st.caption(f"Ensemble visualization for: **{global_eval_class}**")
        
        # Ensemble button
        if st.button("🔗 Generate Ensemble", key="ensemble_btn", use_container_width=True, type="primary"):
            st.session_state['generate_ensemble'] = True
        
        # Generate and display ensemble
        if st.session_state.get('generate_ensemble', False):
            # First, get available classifiers and their confidences
            _, _, classifier_names, confidences = create_ensemble_cam(
                config, selected_image, global_eval_class, transparency_weights=None
            )
            
            if classifier_names:
                # Create sliders for each classifier (collect weights first)
                transparency_weights = {}
                for clf_name in classifier_names:
                    transparency_weights[clf_name] = st.session_state.get(f"transparency_{clf_name}", 50)
                
                # Generate ensemble with transparency weights
                with st.spinner("Creating ensemble CAM..."):
                    ensemble_heatmap, ensemble_overlay, _, _ = create_ensemble_cam(
                        config, selected_image, global_eval_class, 
                        transparency_weights=transparency_weights
                    )
                
                # Display ensemble image FIRST
                if ensemble_heatmap and ensemble_overlay:
                    # Checkbox to toggle between heatmap and overlay
                    show_overlay = st.checkbox("Overlay Image", value=False, key="ensemble_overlay_toggle")
                    
                    # Display selected version
                    if show_overlay:
                        st.image(ensemble_overlay, caption=f"Ensemble CAM (Overlay): {global_eval_class}", use_container_width=True)
                    else:
                        st.image(ensemble_heatmap, caption=f"Ensemble CAM (Heatmap): {global_eval_class}", use_container_width=True)
                    
                    st.divider()
                    
                    # Transparency control table (compact)
                    # Table header
                    cols = st.columns([2, 1, 2])
                    with cols[0]:
                        st.text("Classifier")
                    with cols[1]:
                        st.text("Confidence")
                    with cols[2]:
                        st.text("Transparency")
                    
                    st.divider()
                    
                    # Create a row for each classifier
                    for clf_name in classifier_names:
                        cols = st.columns([2, 1, 2])
                        
                        with cols[0]:
                            st.text(clf_name)
                        
                        with cols[1]:
                            confidence = confidences.get(clf_name, 0.0)
                            st.text(f"{confidence:.1f}%")
                        
                        with cols[2]:
                            transparency = st.slider(
                                f"Transparency {clf_name}",
                                min_value=0,
                                max_value=100,
                                value=50,
                                key=f"transparency_{clf_name}",
                                label_visibility="collapsed",
                                help=f"Control {clf_name}'s contribution"
                            )
                    
                    st.divider()
                    
                    # Show normalized weights
                    total_weight = sum(transparency_weights.values())
                    if total_weight > 0:
                        normalized_weights = {k: (v/total_weight)*100 for k, v in transparency_weights.items()}
                        weight_str = " | ".join([f"{k}: {v:.1f}%" for k, v in normalized_weights.items()])
                        st.caption(f"Normalized weights: {weight_str}")
                    
                    st.info("💡 Adjust sliders to change model contributions. Brighter regions = higher activation")
                else:
                    st.error("Failed to generate ensemble CAM")
            else:
                st.warning("⚠️ No CAM images available. Generate CAMs first by clicking 'Eval'.")
    else:
        st.info("👈 Use Global Eval to generate ensemble CAM")
        st.caption("1. Select a pathology class from dropdown\n2. Click 'Eval' button\n3. Return here and click 'Generate Ensemble'")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Ensemble-CAM",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Add CSS for compact layout and scrollable column 2
    st.markdown("""
        <style>
        /* Hide Streamlit header completely */
        [data-testid="stHeader"] {
            display: none !important;
        }
        
        /* Remove padding - maximize vertical space */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
        /* Reduce overall spacing between elements - COMPACT LAYOUT */
        .element-container {
            margin-bottom: 0.3rem !important;
        }
        
        /* Reduce padding in columns */
        [data-testid="column"] {
            padding: 0.3rem !important;
        }
        
        /* Compact spacing for headers */
        h1 {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        h2 {
            margin-top: 0.3rem !important;
            margin-bottom: 0.3rem !important;
            font-size: 1.3rem !important;
        }
        
        h3 {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
            font-size: 1.1rem !important;
        }
        
        /* Compact button spacing */
        .stButton button {
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
            padding: 0.3rem 0.6rem !important;
        }
        
        /* Compact expander styling - NO SCROLLBARS */
        .streamlit-expanderHeader {
            padding: 0.4rem !important;
        }
        
        .streamlit-expanderContent {
            padding: 0.4rem !important;
            overflow: visible !important;
            max-height: none !important;
        }
        
        /* Remove scrollbars from individual cards */
        [data-testid="stExpander"] {
            overflow: visible !important;
        }
        
        [data-testid="stExpander"] > div {
            overflow: visible !important;
        }
        
        /* Compact divider */
        hr {
            margin: 0.3rem 0 !important;
        }
        
        /* Compact dataframe/table styling */
        [data-testid="stDataFrame"], .dataframe {
            margin: 0.2rem 0 !important;
        }
        
        /* Compact selectbox/dropdown */
        [data-testid="stSelectbox"] {
            margin-bottom: 0.2rem !important;
        }
        
        /* Remove extra spacing in column headers */
        [data-testid="column"] > div {
            gap: 0.2rem !important;
        }
        
        /* FORCE dynamic height on ALL containers - override all defaults */
        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stVerticalBlock"],
        .element-container > div,
        section[data-testid="stVerticalBlock"],
        div.stVerticalBlock {
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
        }
        
        /* Bordered containers (classifier cards) - FORCE compact, no fixed height */
        [data-testid="stVerticalBlockBorderWrapper"] {
            height: fit-content !important;
            min-height: fit-content !important;
            max-height: fit-content !important;
            padding: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* CRITICAL: Remove any fixed heights from st.container(border=True) */
        section[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stVerticalBlockBorderWrapper"][style*="border"],
        .stVerticalBlockBorderWrapper {
            height: auto !important;
            min-height: auto !important;
            max-height: none !important;
        }
        
        /* Compact container spacing */
        .stMarkdown {
            margin-bottom: 0.2rem !important;
        }
        
        /* Remove any min-height from containers */
        [data-testid="stHorizontalBlock"] [data-testid="column"] > div,
        [data-testid="stHorizontalBlock"] section {
            min-height: 0 !important;
        }
        
        /* Target column 2 specifically - make it scrollable with fixed height */
        /* This overrides the general container rules for column 2 only */
        [data-testid="stHorizontalBlock"] > div:nth-child(2) > [data-testid="stVerticalBlock"] {
            height: 92vh !important;
            max-height: 92vh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        
        /* But FORCE cards INSIDE column 2 to be dynamic - most important rule */
        [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stHorizontalBlock"] > div:nth-child(2) section,
        [data-testid="stHorizontalBlock"] > div:nth-child(2) .element-container,
        [data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stVerticalBlock"] {
            height: auto !important;
            min-height: 0 !important;
            max-height: none !important;
            overflow: visible !important;
        }
        
        /* Custom scrollbar for column 2 */
        [data-testid="stHorizontalBlock"] > div:nth-child(2) > [data-testid="stVerticalBlock"]::-webkit-scrollbar {
            width: 8px;
        }
        
        [data-testid="stHorizontalBlock"] > div:nth-child(2) > [data-testid="stVerticalBlock"]::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        [data-testid="stHorizontalBlock"] > div:nth-child(2) > [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        
        [data-testid="stHorizontalBlock"] > div:nth-child(2) > [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        /* Ensure other columns don't have height restrictions */
        [data-testid="stHorizontalBlock"] > div:nth-child(1) > [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"] > div:nth-child(3) > [data-testid="stVerticalBlock"] {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        st.stop()
    
    # Create 3-column layout (Input | Classifiers | Output)
    col1, col2, col3 = st.columns([1, 2, 1.5])
    
    # Column 1: Input (with global evaluation controls)
    with col1:
        selected_image = display_image_selector(config)
    
    # Column 2: Classifiers (with inline inference)
    with col2:
        display_classifiers_panel(config, selected_image)
    
    # Column 3: Output (Ensemble CAM visualization)
    with col3:
        display_output_placeholder(config, selected_image)


if __name__ == "__main__":
    main()

