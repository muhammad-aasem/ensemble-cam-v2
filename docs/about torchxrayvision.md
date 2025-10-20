# About TorchXRayVision

TorchXRayVision is a comprehensive PyTorch-based library for chest X-ray medical imaging research, providing a unified framework for working with multiple chest X-ray datasets (including NIH, CheXpert, MIMIC, RSNA Pneumonia, SIIM Pneumothorax, and others) along with pre-trained models for classification, segmentation, and autoencoding tasks. 

## Overview

The library offers over 18 different pathology labels with standardized alignment across datasets, includes pre-trained DenseNet and ResNet models, and provides specialized tools for distribution shift analysis, pathology mask extraction, data augmentation, and covariate dataset manipulation to study model generalization. 

Developed by researchers at MILA Quebec AI Institute, Stanford's Center for AI in Medicine & Imaging, and supported by CIFAR, the library was designed to facilitate reproducible research in medical AI, particularly for studying cross-domain generalization in automated X-ray prediction, with all components accessible through a simple pip-installable package that handles dataset downloading, preprocessing, and model loading with minimal configuration.

## Classification

TorchXRayVision provides multiple pre-trained classification models for chest X-ray pathology detection. All models use deep convolutional neural networks trained on large-scale medical imaging datasets.

### Available Classification Models

#### DenseNet Models (Recommended)

| Model Weight | Architecture | Input Size | Pathologies | Training Dataset(s) | Model Size | Performance |
|-------------|--------------|------------|-------------|-------------------|------------|-------------|
| **densenet121-res224-all** ⭐ | DenseNet121 | 224×224 | **18** | NIH + PadChest + CheXpert + MIMIC-CXR + Google + OpenI + Kaggle | ~27MB | **Best overall** - Highest generalization |
| **densenet121-res224-nih** | DenseNet121 | 224×224 | 14 | NIH ChestX-ray14 | ~27MB | Excellent on NIH test set |
| **densenet121-res224-pc** | DenseNet121 | 224×224 | 14 | PadChest | ~27MB | Best for European X-rays |
| **densenet121-res224-chex** | DenseNet121 | 224×224 | 10 | CheXpert | ~27MB | Optimized for CheXpert labels |
| **densenet121-res224-mimic_nb** | DenseNet121 | 224×224 | 10 | MIMIC-CXR (frontal) | ~27MB | Frontal view specialist |
| **densenet121-res224-mimic_ch** | DenseNet121 | 224×224 | 10 | MIMIC-CXR (chest) | ~27MB | All chest view types |
| **densenet121-res224-rsna** | DenseNet121 | 224×224 | 2 | RSNA Pneumonia Challenge | ~27MB | Pneumonia specialist |

#### ResNet Models (Alternative Architecture)

| Model Weight | Architecture | Input Size | Pathologies | Training Dataset(s) | Model Size | Performance |
|-------------|--------------|------------|-------------|-------------------|------------|-------------|
| **resnet50-res512-all** | ResNet50 | 512×512 | 18 | Combined datasets | ~98MB | High resolution, more detail |

**Note:** ResNet50 uses 512×512 input resolution (vs 224×224 for DenseNet), providing more spatial detail but slower inference.

#### Baseline Models (Specialized)

| Model | Architecture | Input Size | Pathologies | Source | Model Size | Performance |
|-------|--------------|------------|-------------|--------|------------|-------------|
| **JF Healthcare DenseNet** | DenseNet121 | 512×512 | **5** | JF Healthcare | ~28MB | Specialized for 5 key pathologies |

**Note:** Baseline models are specialized versions focusing on specific pathology sets. The JF Healthcare model detects: **Cardiomegaly, Edema, Consolidation, Atelectasis, Effusion**.

### Model Capabilities by Pathology

All models output probability scores (0-1) for each supported pathology. The comprehensive pathology set (18 total) includes:

| Pathology | All | NIH | PC | CheX | MIMIC | RSNA | Clinical Significance |
|-----------|:---:|:---:|:--:|:----:|:-----:|:----:|----------------------|
| **Atelectasis** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | Lung collapse or closure |
| **Consolidation** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | Lung tissue filled with liquid |
| **Infiltration** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Substance denser than air in lungs |
| **Pneumothorax** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | Collapsed lung |
| **Edema** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | Fluid in lungs |
| **Emphysema** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Damaged air sacs |
| **Fibrosis** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Lung tissue scarring |
| **Effusion** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | Fluid around lungs |
| **Pneumonia** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Lung infection/inflammation |
| **Pleural Thickening** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Thickening of pleura |
| **Cardiomegaly** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | Enlarged heart |
| **Nodule** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Small round growth |
| **Mass** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Larger growth/lesion |
| **Hernia** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | Hiatal hernia |
| **Lung Lesion** | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Abnormal lung tissue |
| **Fracture** | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | Bone fracture |
| **Lung Opacity** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | General opacity/cloudiness |
| **Enlarged Cardiomediastinum** | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | Enlarged heart/mediastinum |

### Performance Metrics

Performance varies by dataset and pathology. Here are representative AUROC (Area Under ROC Curve) scores:

#### DenseNet121-All Model Performance (Representative)

| Pathology | NIH Test | CheXpert Test | MIMIC Test | Notes |
|-----------|----------|---------------|------------|-------|
| Atelectasis | 0.79-0.82 | 0.81-0.84 | 0.80-0.83 | Good consistency |
| Cardiomegaly | 0.88-0.91 | 0.87-0.90 | 0.89-0.92 | High accuracy |
| Consolidation | 0.78-0.81 | 0.79-0.82 | 0.80-0.83 | Consistent |
| Edema | 0.88-0.91 | 0.89-0.92 | 0.90-0.93 | High accuracy |
| Effusion | 0.86-0.89 | 0.87-0.90 | 0.88-0.91 | Very reliable |
| Pneumonia | 0.73-0.76 | 0.77-0.80 | 0.76-0.79 | Moderate |
| Pneumothorax | 0.86-0.89 | 0.88-0.91 | 0.87-0.90 | High accuracy |

**Note:** Performance metrics are approximate and depend on specific test sets, thresholds, and evaluation protocols.

### Model Selection Guidelines

**For Maximum Coverage:** Use `densenet121-res224-all` (18 pathologies)
- Best for general-purpose screening
- Highest cross-dataset generalization
- Trained on most diverse data

**For Specific Datasets:** Use dataset-matched weights
- `nih`: Best benchmark compatibility with NIH
- `chex`: Optimized for CheXpert labels
- `mimic`: Best for US hospital data

**For Specialized Tasks:** 
- `rsna`: Pneumonia detection only (fastest, 2 pathologies)
- `resnet50`: Higher resolution 512×512 input (slower but more spatial detail)

**For Research:** Choose based on evaluation dataset
- Ensures fair comparison with published benchmarks
- Maintains label alignment consistency

### Technical Specifications

**All Classification Models:**
- **Framework:** PyTorch
- **Input Format:** Single-channel (grayscale) or RGB
- **Normalization:** Zero-centered with dataset-specific statistics
- **Output:** Sigmoid probabilities for multi-label classification
- **Inference Time:** ~50-200ms per image (CPU), ~10-30ms (GPU)
- **Memory Requirements:** 
  - Model: 27MB (DenseNet121) to 98MB (ResNet50)
  - Runtime: ~2GB RAM (DenseNet), ~4GB RAM (ResNet50), ~1-2GB GPU VRAM

**Training Details:**
- **Augmentation:** Random rotation (±45°), translation (±15%), scaling (±15%)
- **Optimizer:** Adam with weight decay
- **Loss Function:** Binary cross-entropy with label smoothing
- **Batch Size:** 16-32 per GPU
- **Training Time:** 2-7 days on 4×V100 GPUs

### Model Provenance

All pre-trained models in TorchXRayVision are:
- ✅ **Peer-reviewed:** Published in academic venues
- ✅ **Open source:** Apache 2.0 license
- ✅ **Reproducible:** Training code available
- ✅ **Benchmarked:** Evaluated on standard test sets
- ✅ **Version controlled:** Tagged releases on GitHub

**Citation:** Cohen et al. "TorchXRayVision: A library of chest X-ray datasets and models." Medical Imaging with Deep Learning, 2022.

### Usage in This Project

Three classifier scripts are available:

**DenseNet121 Classifier** (224×224, faster, 18 pathologies, recommended):
```bash
# Default: densenet121-res224-all (18 pathologies)
uv run python src/classify_densenet121.py --input xray.jpg

# Specify different weights
uv run python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-nih

# Adjust threshold for sensitivity/specificity tradeoff
uv run python src/classify_densenet121.py --input xray.jpg --threshold 0.7
```

**ResNet50 Classifier** (512×512, high resolution, 18 pathologies):
```bash
# High resolution model (slower but more spatial detail)
uv run python src/classify_resnet50.py --input xray.jpg

# With custom threshold
uv run python src/classify_resnet50.py --input xray.jpg --threshold 0.6
```

**JF Healthcare Baseline Classifier** (512×512, specialized, 5 pathologies):
```bash
# Specialized baseline model focusing on 5 key pathologies
uv run python src/classify_jfhealthcare.py --input xray.jpg

# Cardiomegaly, Edema, Consolidation, Atelectasis, Effusion only
uv run python src/classify_jfhealthcare.py --input xray.jpg --threshold 0.6
```

For detailed model comparison and selection guide, see [Model Comparison Documentation](model-comparison.md).

## Dataset Support

TorchXRayVision integrates with the following chest X-ray datasets:

### Public Datasets Available

| Dataset | Images | Pathologies | Source | Access |
|---------|--------|-------------|--------|--------|
| **NIH ChestX-ray14** | 112,120 | 14 | NIH Clinical Center | Public |
| **CheXpert** | 224,316 | 14 | Stanford | Registration required |
| **MIMIC-CXR** | 377,110 | 14 | MIT/Beth Israel | PhysioNet credentialing |
| **PadChest** | 160,000 | 174 | Hospital San Juan | Registration required |
| **RSNA Pneumonia** | 26,684 | 1 | RSNA/Kaggle | Public |
| **SIIM Pneumothorax** | 12,047 | 1 + masks | SIIM/Kaggle | Public |
| **OpenI** | 7,470 | 14 | NIH Open-i | Public |
| **COVID-19 Image** | ~5,000+ | Multiple | GitHub crowd-sourced | Public |

### Dataset Features

- **Unified API:** Same interface across all datasets
- **Automatic Download:** Most datasets download automatically
- **Standardized Labels:** Pathology names aligned across datasets
- **Metadata Access:** Age, sex, view position, etc.
- **Masks Available:** For pneumothorax, lung opacity, cardiomegaly

## References

**Primary Paper:**
```
Cohen, J.P., Viviano, J.D., Bertin, P., Morrison, P., Torabian, P., Guarrera, M., 
Lungren, M.P., Chaudhari, A., Brooks, R., Hashir, M. and Bertrand, H., 2022.
TorchXRayVision: A library of chest X-ray datasets and models. 
Medical Imaging with Deep Learning.
```

**Key Publications:**
- Cross-domain generalization: arXiv:2002.02497
- Multi-task learning: Various MICCAI/MIDL workshops
- Dataset alignment: Medical Image Analysis

**Links:**
- GitHub: https://github.com/mlmed/torchxrayvision
- Documentation: https://mlmed.org/torchxrayvision
- Paper: https://arxiv.org/abs/2111.00595

---

*This project uses TorchXRayVision v1.4.0 with UV package management for reproducible environments.*
