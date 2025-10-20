# PyTorch Grad-CAM: Visual Explainability for X-ray Classifiers

## Overview

[PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) is an advanced AI explainability library that provides visual explanations for deep learning models in computer vision. It generates **Class Activation Maps (CAMs)** - heatmaps that highlight which regions of an input image were most influential in a model's prediction.

**Key Features:**
- ✅ Support for CNNs and Vision Transformers
- ✅ Multiple CAM methods (GradCAM, GradCAM++, ScoreCAM, etc.)
- ✅ Works with classification, object detection, and segmentation
- ✅ Noise reduction through smoothing techniques
- ✅ Metrics for evaluating explanation quality
- ✅ Pure PyTorch implementation

**Repository:** https://github.com/jacobgil/pytorch-grad-cam  
**License:** MIT

---

## Why Use Grad-CAM for Medical X-ray Analysis?

In medical imaging, **interpretability is critical**. Grad-CAM helps answer:

1. **"Why did the model predict Pneumonia?"**  
   → Shows lung regions with infiltrates that influenced the prediction

2. **"Is the model focusing on the right areas?"**  
   → Validates that the model looks at anatomically relevant regions

3. **"How confident should we be in this prediction?"**  
   → Diffuse vs. focused heatmaps indicate certainty levels

4. **"Are there multiple pathologies?"**  
   → Different heatmaps for different predicted conditions

### Clinical Relevance

- **Trust Building**: Radiologists can verify the model's reasoning
- **Error Detection**: Identify when models focus on artifacts or non-clinical features
- **Education**: Help medical students understand pathology patterns
- **Regulatory Compliance**: Explainability required for FDA approval (AI/ML in medical devices)

---

## Available CAM Methods

The library provides multiple CAM algorithms, each with different characteristics:

| Method | Speed | Quality | Best For |
|--------|-------|---------|----------|
| **GradCAM** | ⚡⚡⚡ Fast | Good | General use, real-time |
| **GradCAM++** ⭐ | ⚡⚡ Moderate | Excellent | **Recommended for X-rays** |
| **HiResCAM** | ⚡⚡ Moderate | Excellent | High-resolution details |
| **ScoreCAM** | ⚡ Slow | Very Good | Gradient-free, robust |
| **AblationCAM** | ⚡ Slow | Very Good | Most faithful explanations |
| **LayerCAM** | ⚡⚡ Moderate | Good | Multi-layer analysis |
| **EigenCAM** | ⚡⚡⚡ Fast | Good | Quick PCA-based |
| **XGradCAM** | ⚡⚡⚡ Fast | Good | Improved GradCAM |

### Why GradCAM++ for Medical Images?

**GradCAM++** is recommended for X-ray classification because:
- ✅ Better localization for multiple instances (e.g., bilateral pneumonia)
- ✅ Improved pixel-level explanations vs. original GradCAM
- ✅ Better handling of small objects (nodules, lesions)
- ✅ Fast enough for real-time clinical use
- ✅ Widely validated in medical imaging research

**Reference:** [Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/abs/1710.11063)

---

## Integration with TorchXRayVision

### Compatibility

PyTorch Grad-CAM works seamlessly with TorchXRayVision models because:
1. **Both use PyTorch**: No framework conversion needed
2. **Standard CNN architectures**: DenseNet, ResNet are fully supported
3. **Single forward pass**: Compatible with torchxrayvision preprocessing
4. **Batch processing**: Can generate CAMs for multiple images

### Supported TorchXRayVision Models

| Model | Architecture | CAM Support | Notes |
|-------|-------------|-------------|-------|
| DenseNet121 | DenseNet | ✅ Full | Target `model.features` |
| ResNet50 | ResNet | ✅ Full | Target `model.layer4` |
| JF Healthcare | DenseNet | ✅ Full | Target `model.features` |

---

## Implementation Guide

### 1. Installation

```bash
# Add to project dependencies
uv add grad-cam
```

Or update `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies
    "grad-cam>=1.5.0",
]
```

### 2. Basic Usage with TorchXRayVision

```python
import torch
import torchxrayvision as xrv
import numpy as np
import cv2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Load TorchXRayVision model
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

# Preprocess X-ray image (using torchxrayvision's method)
from skimage.io import imread
img = imread("xray.jpg")
img = xrv.datasets.normalize(img, maxval=255)
input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

# Define target layer (last convolutional layer)
target_layers = [model.features[-1]]

# Initialize GradCAM++
cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

# Specify target class (e.g., Pneumonia = index 10)
targets = [ClassifierOutputTarget(10)]

# Generate CAM
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# Overlay on original image
cam_image = show_cam_on_image(
    img.astype(np.float32) / 255,  # Original image normalized
    grayscale_cam[0, :],           # CAM heatmap
    use_rgb=True                    # Output RGB image
)
```

### 3. Integration with Ensemble-CAM Web UI

For the current project, CAMs can be added to the Output panel:

```python
# In app.py, add after prediction

def generate_cam_for_model(model, image_tensor, target_class_idx, model_type):
    """Generate CAM heatmap for a specific pathology prediction."""
    
    # Define target layer based on model type
    if model_type == "densenet":
        target_layers = [model.features[-1]]
    elif model_type == "resnet":
        target_layers = [model.layer4[-1]]
    elif model_type == "jfhealthcare":
        target_layers = [model.features[-1]]
    
    # Initialize GradCAM++
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    # Target the predicted class
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    
    return grayscale_cam[0]

# In the classify button handler:
if classify_button:
    for clf_key, clf_data in selected_classifiers.items():
        # ... existing prediction code ...
        
        # Get top prediction index
        top_pathology = max(results, key=results.get)
        top_class_idx = model.pathologies.index(top_pathology)
        
        # Generate CAM
        cam_heatmap = generate_cam_for_model(
            model, 
            img_tensor, 
            top_class_idx,
            clf_config['model_type']
        )
        
        # Store CAM for display
        st.session_state.cams[clf_config['name']] = cam_heatmap
```

### 4. Displaying Multiple CAMs

For multi-pathology visualization:

```python
import matplotlib.pyplot as plt

def visualize_top_predictions_with_cams(model, image_tensor, results, top_k=3):
    """Generate CAMs for top-K predicted pathologies."""
    
    target_layers = [model.features[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    
    # Get top-K pathologies
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 4))
    
    # Show original image
    axes[0].imshow(image_tensor.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title("Original X-ray")
    axes[0].axis('off')
    
    # Generate CAM for each top prediction
    for idx, (pathology, prob) in enumerate(sorted_results):
        class_idx = model.pathologies.index(pathology)
        targets = [ClassifierOutputTarget(class_idx)]
        
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        
        # Display heatmap
        axes[idx + 1].imshow(image_tensor.squeeze().cpu().numpy(), cmap='gray')
        axes[idx + 1].imshow(grayscale_cam[0], alpha=0.5, cmap='jet')
        axes[idx + 1].set_title(f"{pathology}\n{prob:.2%}")
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    return fig
```

---

## Advanced Features

### 1. Smoothing for Better Visualizations

Reduce noise and improve CAM quality:

```python
# Test-time augmentation (6x slower, better quality)
cam = GradCAMPlusPlus(
    model=model, 
    target_layers=target_layers,
    use_cuda=torch.cuda.is_available()
)

# Generate smooth CAM
grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets,
    aug_smooth=True,      # Test-time augmentation
    eigen_smooth=True     # PCA smoothing
)
```

**Results:**
- `aug_smooth=True`: Better centering on pathology regions
- `eigen_smooth=True`: Removes high-frequency noise
- Combined: Clinical-grade visualization quality

### 2. Guided Backpropagation

Enhance detail in explanations:

```python
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import deprocess_image

# Generate guided backprop
gb_model = GuidedBackpropReLUModel(model=model, device='cuda')
gb = gb_model(input_tensor, target_category=10)

# Combine with CAM
cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
cam_gb = deprocess_image(cam_mask * gb)  # Guided Grad-CAM
```

### 3. Quantitative Evaluation

Measure explanation quality:

```python
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget

# Evaluate CAM faithfulness
metric_target = ClassifierOutputSoftmaxTarget(target_class_idx)
cam_metric = ROADMostRelevantFirst(percentile=75)

scores = cam_metric(
    input_tensor, 
    grayscale_cam, 
    [metric_target], 
    model
)

print(f"Explanation Faithfulness Score: {scores[0]:.4f}")
```

**Metrics Available:**
- **ROAD (Remove And Debias)**: State-of-the-art faithfulness metric
- **Confidence Drop**: Measures prediction change when removing important regions
- **Average Drop**: Average probability drop across perturbations

---

## Best Practices for Medical Imaging

### 1. Choose Appropriate Target Layers

```python
# DenseNet121
target_layers = [model.features[-1]]  # or model.features.denseblock4

# ResNet50
target_layers = [model.layer4[-1]]  # or model.layer4[2]

# Multiple layers for hierarchical analysis
target_layers = [model.layer3[-1], model.layer4[-1]]
```

### 2. Generate CAMs for Multiple Pathologies

```python
# Get predictions for all pathologies
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.sigmoid(outputs)[0]

# Generate CAM only for significant predictions (prob > threshold)
threshold = 0.5
cams = {}

for idx, (pathology, prob) in enumerate(zip(model.pathologies, probs)):
    if prob > threshold:
        targets = [ClassifierOutputTarget(idx)]
        cam_heatmap = cam(input_tensor=input_tensor, targets=targets)
        cams[pathology] = cam_heatmap[0]
```

### 3. Consistent Preprocessing

```python
def preprocess_for_cam(image_path, target_size=224):
    """
    Preprocess X-ray for CAM generation.
    Must match the preprocessing used during training.
    """
    from skimage.io import imread
    from skimage.transform import resize
    
    # Load and convert to grayscale
    img = imread(image_path)
    if len(img.shape) > 2:
        img = img.mean(axis=2)
    
    # Resize
    img = resize(img, (target_size, target_size), 
                 anti_aliasing=True, preserve_range=True)
    
    # Normalize using torchxrayvision method
    img = xrv.datasets.normalize(img, maxval=255, reshape=False)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    
    return img_tensor, img  # Return both for visualization
```

### 4. Save High-Quality Outputs

```python
import matplotlib.pyplot as plt

def save_cam_visualization(original_img, cam_heatmap, pathology, 
                           probability, output_path):
    """Save CAM overlay as high-resolution image."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original X-ray
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original X-ray')
    axes[0].axis('off')
    
    # CAM heatmap only
    axes[1].imshow(cam_heatmap, cmap='jet')
    axes[1].set_title(f'CAM Heatmap\n{pathology}')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_img, cmap='gray')
    axes[2].imshow(cam_heatmap, alpha=0.5, cmap='jet')
    axes[2].set_title(f'Overlay\nProbability: {probability:.2%}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## Performance Considerations

### Speed Comparison (224×224 image, CPU)

| Method | Time | Relative |
|--------|------|----------|
| GradCAM | 0.05s | 1× |
| **GradCAM++** | **0.08s** | **1.6×** |
| HiResCAM | 0.10s | 2× |
| ScoreCAM | 2.5s | 50× |
| AblationCAM | 3.0s | 60× |

**Recommendation for Real-Time Clinical Use:**
- **Interactive UI**: GradCAM++, GradCAM, or XGradCAM
- **Batch Processing**: Any method (pre-compute overnight)
- **Research Analysis**: ScoreCAM or AblationCAM for highest quality

### Memory Usage

- **Single CAM**: ~50MB additional RAM
- **Smoothing enabled**: ~300MB (6 forward passes)
- **Batch processing**: Linear scaling

### GPU Acceleration

```python
# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
input_tensor = input_tensor.to(device)

cam = GradCAMPlusPlus(
    model=model,
    target_layers=target_layers,
    use_cuda=(device == 'cuda')
)
```

**GPU Speedup:** 10-20× faster for batch processing

---

## Troubleshooting

### Issue: CAM is all uniform/no localization

**Cause:** Wrong target layer selected

**Solution:**
```python
# Print all layers
for name, module in model.named_modules():
    print(name, type(module))

# Use the last convolutional layer
# For DenseNet: model.features[-1]
# For ResNet: model.layer4[-1]
```

### Issue: CAM highlights wrong regions

**Cause:** Model might be focusing on artifacts (e.g., text labels, borders)

**Solution:**
- Validate model predictions first
- Try different CAM methods (GradCAM++ vs. ScoreCAM)
- Check if preprocessing matches training

### Issue: Memory error

**Cause:** Smoothing or batch size too large

**Solution:**
```python
# Reduce batch size
cam.batch_size = 16  # Default is 32

# Disable smoothing for large images
grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets,
    aug_smooth=False,    # Disable augmentation
    eigen_smooth=False   # Disable PCA
)
```

---

## Example Output

### Single Pathology (Pneumonia)

```
Input: chest_xray.jpg (2048×2048)
Model: DenseNet121-all
Prediction: Pneumonia (87.3%)

CAM Visualization:
- Red regions: High influence (consolidation in lower right lobe)
- Yellow regions: Moderate influence (perihilar infiltrates)
- Blue regions: Low influence (heart, diaphragm)
```

### Multi-Pathology (Pneumonia + Effusion)

```
Top-3 Predictions with CAMs:

1. Pneumonia (87.3%)
   → CAM highlights: Lower right lobe consolidation

2. Effusion (72.1%)
   → CAM highlights: Costophrenic angle blunting

3. Infiltration (54.2%)
   → CAM highlights: Bilateral lower zones
```

---

## Clinical Validation Workflow

1. **Generate predictions** with TorchXRayVision classifier
2. **Create CAMs** for top-K pathologies (K=3 recommended)
3. **Radiologist review**: Verify CAM alignment with clinical findings
4. **Quality scoring**: Use ROAD metrics for explanation faithfulness
5. **Documentation**: Save CAMs alongside predictions for audit trail

---

## Future Enhancements for Ensemble-CAM

Planned features integrating Grad-CAM:

- [ ] **Real-time CAM generation** in Output panel
- [ ] **Side-by-side comparison** of CAMs from 3 classifiers
- [ ] **Ensemble CAM**: Average CAMs from multiple models
- [ ] **Interactive threshold**: Adjust CAM sensitivity
- [ ] **Export CAMs**: Save as PNG with annotations
- [ ] **Batch CAM generation**: Process multiple images
- [ ] **CAM metrics dashboard**: Faithfulness scores
- [ ] **Differential CAMs**: Compare before/after treatment

---

## References

### Primary Papers

1. **Grad-CAM** (2017): [Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
2. **Grad-CAM++** (2018): [Improved Visual Explanations for Deep Convolutional Networks](https://arxiv.org/abs/1710.11063)
3. **HiResCAM** (2020): [Use HiResCAM instead of Grad-CAM for faithful explanations](https://arxiv.org/abs/2011.08891)
4. **Score-CAM** (2020): [Score-Weighted Visual Explanations for CNNs](https://arxiv.org/abs/1910.01279)

### Medical Imaging Applications

- **COVID-19 Detection**: CAMs validate model focus on ground-glass opacities
- **Tuberculosis Screening**: Localize cavitations and infiltrates
- **Pneumonia Classification**: Distinguish bacterial vs. viral patterns
- **Cardiomegaly**: Verify cardiac silhouette measurements

### Library Resources

- **GitHub**: https://github.com/jacobgil/pytorch-grad-cam
- **Documentation**: https://jacobgil.github.io/pytorch-gradcam-book
- **PyPI**: `pip install grad-cam`
- **License**: MIT (commercial use allowed)

---

## Next Steps

To implement CAM visualization in the Ensemble-CAM project:

1. **Add dependency**: `uv add grad-cam`
2. **Create CAM utilities**: `src/cam_utils.py`
3. **Update web UI**: Implement Output panel with CAM display
4. **Add configuration**: CAM method selection in `config_classifiers.toml`
5. **Testing**: Validate CAMs align with clinical expectations

See [WEB_UI_GUIDE.md](WEB_UI_GUIDE.md#adding-cam-visualization) for implementation details.

