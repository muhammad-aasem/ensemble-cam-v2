# Ensemble-CAM2: Weak-supervised Localization of thoracic diseases in X-rays images with interpretability

A comprehensive chest X-ray analysis system, featuring multi-model classification, CAM (Class Activation Map) visualization, and automated pathology localization with bounding box generation and evaluation metrics.

## Key Features

- Four Classification Models - DenseNet121, ResNet50, JF Healthcare baseline, and MobileNetV3
- Custom Model Training - Train your own MobileNetV3 model on NIH Chest X-ray14 dataset
- CAM Visualization - Generate heatmaps to visualize model predictions
- Automated Localization - Generate bounding boxes from CAMs with evaluation metrics
- Multi-Model Ensemble - Combine predictions from multiple models
- Comprehensive Metrics - IoU, Dice, ColLoc, mAP for quantitative evaluation
- CLI Tools - Complete command-line pipeline for batch processing
- Complete Pipeline - From training to classification to CAM to bounding box to evaluation

## Quick Start

### Prerequisites

- [UV](https://docs.astral.sh/uv/) package manager installed
- Python 3.12 (automatically managed by UV)

### Installation

All dependencies are managed by UV and specified in `pyproject.toml`. To install:

```bash
uv sync
```

### Usage

#### Command Line Interface

Run classifiers individually from the terminal:

```bash
# DenseNet121 (default, 224×224, faster, 18 pathologies)
uv run python src/classify_densenet121.py --input path/to/xray.jpg

# ResNet50 (512×512, high resolution, 18 pathologies)
uv run python src/classify_resnet50.py --input path/to/xray.jpg

# JF Healthcare Baseline (512×512, 5 key pathologies)
uv run python src/classify_jfhealthcare.py --input path/to/xray.jpg

# MobileNetV3 (224×224, custom trained, 14 pathologies - NIH)
uv run python src/classify_mobilenetv3.py --input path/to/xray.jpg
```

#### DenseNet121 Classifier Options

```bash
uv run python src/classify_densenet121.py \
    --input path/to/xray.jpg \
    --threshold 0.6 \
    --weights densenet121-res224-all \
    --device cuda
```

**Arguments:**
- `--input` (required): Path to X-ray image (JPEG, PNG, etc.)
- `--threshold` (optional): Prediction threshold (default: 0.5)
- `--weights` (optional): DenseNet121 weights (default: densenet121-res224-all)
  - densenet121-res224-all, densenet121-res224-nih, densenet121-res224-pc,
  - densenet121-res224-chex, densenet121-res224-mimic_nb/ch, densenet121-res224-rsna
- `--device` (optional): Device to use - `cuda`, `cpu`, or `auto` (default: auto)

#### ResNet50 Classifier Options

```bash
uv run python src/classify_resnet50.py \
    --input path/to/xray.jpg \
    --threshold 0.6 \
    --device cuda
```

**Arguments:**
- `--input` (required): Path to X-ray image (JPEG, PNG, etc.)
- `--threshold` (optional): Prediction threshold (default: 0.5)
- `--weights` (optional): ResNet50 weights (default: resnet50-res512-all)
- `--device` (optional): Device to use - `cuda`, `cpu`, or `auto` (default: auto)

**Note:** ResNet50 processes 512×512 images (4× more pixels) for higher spatial detail but is ~3-4× slower than DenseNet121.

#### JF Healthcare Baseline Classifier Options

```bash
uv run python src/classify_jfhealthcare.py \
    --input path/to/xray.jpg \
    --threshold 0.6 \
    --device cuda
```

**Arguments:**
- `--input` (required): Path to X-ray image (JPEG, PNG, etc.)
- `--threshold` (optional): Prediction threshold (default: 0.5)
- `--device` (optional): Device to use - `cuda`, `cpu`, or `auto` (default: auto)

**Note:** This baseline model from JF Healthcare detects **5 key pathologies only**: Cardiomegaly, Edema, Consolidation, Atelectasis, Effusion. For comprehensive detection, use DenseNet121 or ResNet50.

#### Training Your Own MobileNetV3 Model

Train a custom MobileNetV3 model on NIH Chest X-ray14 dataset:

```bash
# Download NIH Chest X-ray14 dataset first
# Then train:
uv run python src/train_mobilenet.py \
    --epochs 50 \
    --imgpath /path/to/nih/images \
    --csvpath /path/to/Data_Entry_2017.csv \
    --batch-size 16 \
    --lr 0.001

# Resume training (automatically picks up from best checkpoint):
uv run python src/train_mobilenet.py --epochs 50 --imgpath /path/to/images --csvpath /path/to/csv

# Use the trained model:
uv run python src/classify_mobilenetv3.py --input path/to/xray.jpg
```

**Training Features:**
- Automatic checkpoint saving (best and last)
- Resume from last best checkpoint
- Performance logging to CSV (`weights/train_weight_mobilenet/performance.csv`)
- Learning rate scheduling
- Training and validation metrics (loss, AUC)
- All artifacts saved in `weights/train_weight_mobilenet/`

#### CAM Generation & Analysis (Visual Explainability)

Generate **Class Activation Maps** to visualize which regions of the X-ray influenced the model's predictions:

##### 1. Generate CAMs (Heatmaps)

```bash
# Generate CAM for specific pathology
uv run python src/generate_cams.py \
    --input data/INPUT/xray.jpg \
    --model densenet121 \
    --target-class "Cardiomegaly"

# Generate for all top predictions
uv run python src/generate_cams.py \
    --input data/INPUT/xray.jpg \
    --model densenet121 \
    --top-k 3

# Use with MobileNetV3 (custom trained)
uv run python src/generate_cams.py \
    --input data/INPUT/xray.jpg \
    --model mobilenetv3 \
    --target-class "Cardiomegaly"
```

**Available Models:** densenet121, resnet50, jfhealthcare, mobilenetv3

**Output Files (per pathology):**
- `{model}_{pathology}_P0.png` - Overlay (CAM on X-ray)
- `{model}_{pathology}_P1.png` - Heatmap only
- `{model}_{pathology}_P2.png` - Red channel extraction
- `{model}_{pathology}_P3.png` - Binary mask
- `_input_image_.png` - Copy of input X-ray
- `metadata.json` - All predictions and metadata

##### 2. Generate Bounding Boxes

```bash
# Generate bounding box from binary mask
uv run python src/generate_bbox.py \
    --input-mask OUTPUT/00005066_030/densenet121_Cardiomegaly_P3.png \
    --ground-truth "Cardiomegaly,277,459,540,301"
```

**Output Files:**
- `{model}_{pathology}_PBBox.json` - Bbox coordinates + metrics (IoU, Dice, ColLoc, mAP)
- `{model}_{pathology}_PBBox.png` - Visualization on X-ray (green=predicted, red=ground truth)

**Metrics Computed:**
- IoU (Intersection over Union)
- Dice Coefficient
- Pixel Accuracy
- Precision@0.5, Recall@0.5
- ColLoc@0.1, ColLoc@0.3 (Correct Localization)
- mAP (mean Average Precision)

##### 3. Consolidated Multi-Model Bounding Boxes

```bash
# Merge masks from multiple models
uv run python src/generate_bbox_consolidated.py \
    --input-sources \
        OUTPUT/00005066_030/densenet121_Cardiomegaly_P3.png \
        OUTPUT/00005066_030/jfhealthcare_Cardiomegaly_P3.png \
        OUTPUT/00005066_030/resnet50_Cardiomegaly_P3.png \
    --method merge-masking \
    --target-class "Cardiomegaly" \
    --ground-truth "Cardiomegaly,277,459,540,301"
```

**Output Files:**
- `{pathology}_consolidated.png` - Merged mask from all models
- `{pathology}_consolidated_bbox.json` - Consolidated bbox + metrics
- `{pathology}_consolidated_bbox.png` - Visualization with all bboxes

**Complete Workflow Example:**
```bash
# Step 1: Classify
python src/classify_densenet121.py --input xray.jpg

# Step 2: Generate CAMs (creates P0, P1, P2, P3)
python src/generate_cams.py --input xray.jpg --model densenet121 --target-class "Cardiomegaly"

# Step 3: Generate bounding box with evaluation
python src/generate_bbox.py --input-mask OUTPUT/xray/densenet121_Cardiomegaly_P3.png \
    --ground-truth "Cardiomegaly,277,459,540,301"

# Step 4: Consolidate multiple models (optional)
python src/generate_bbox_consolidated.py \
    --input-sources OUTPUT/xray/*_Cardiomegaly_P3.png \
    --method merge-masking \
    --target-class "Cardiomegaly" \
    --ground-truth "Cardiomegaly,277,459,540,301"
```

See [docs/about pytorchgradcam.md](docs/about%20pytorchgradcam.md) for detailed CAM documentation.

### Example Output

```
Loading model on cpu...
Processing image: xray.jpg
Generating predictions...

============================================================
X-RAY CLASSIFICATION RESULTS
============================================================

All Predictions (sorted by probability):
------------------------------------------------------------
Lung Opacity                  : 0.8234 (POSITIVE)
Pneumonia                     : 0.6789 (POSITIVE)
Effusion                      : 0.5423 (POSITIVE)
Atelectasis                   : 0.4567 (negative)
...

Positive Findings (≥0.5):
  • Lung Opacity: 0.8234
  • Pneumonia: 0.6789
  • Effusion: 0.5423
============================================================
```

## Documentation

- [CAM Generation Guide](docs/about%20pytorchgradcam.md) - Visual explainability with Grad-CAM
- [Setup Guide](docs/setup.md) - Detailed setup and troubleshooting
- [Model Comparison](docs/model-comparison.md) - Which model weight to use and when
- [About TorchXRayVision](docs/about%20torchxrayvision.md) - Library overview, including baseline models
- [Weights Documentation](weights/README.md) - Model weights management and pathology coverage

## Models

Four classifier scripts are available with different architectures and specializations:

### 1. DenseNet121 Classifier (Recommended)

**Multiple pre-trained weight options:**

| Weight Name | Pathologies | Training Dataset(s) | Best For |
|------------|-------------|---------------------|----------|
| `densenet121-res224-all` (Default) | **18** | NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI, Kaggle | **General use** |
| `densenet121-res224-nih` | 14 | NIH ChestX-ray14 | NIH dataset compatibility |
| `densenet121-res224-pc` | 14 | PadChest | European dataset |
| `densenet121-res224-chex` | 10 | CheXpert | Stanford dataset |
| `densenet121-res224-mimic_nb` | 10 | MIMIC-CXR (frontal) | US hospital frontal views |
| `densenet121-res224-mimic_ch` | 10 | MIMIC-CXR (chest) | US hospital chest views |
| `densenet121-res224-rsna` | 2 | RSNA Pneumonia Challenge | Pneumonia detection only |
| `resnet50-res512-all` | 18 | NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI, Kaggle | High resolution (512×512) |

**Default:** `densenet121-res224-all`
- Most comprehensive: **18 pathologies**
- Input resolution: 224×224 pixels (DenseNet) or 512×512 pixels (ResNet50)
- Trained on all datasets combined (most robust)

### 2. ResNet50 Classifier (High Resolution)

**Alternative Architecture:** `resnet50-res512-all`
- Higher resolution: 512×512 input (vs 224×224 for DenseNet)
- Same 18 pathologies
- More spatial detail but slower inference (~3-4x slower)

### 3. JF Healthcare Baseline Classifier (Specialized)

**Specialized Baseline Model:**
- Architecture: DenseNet121
- Input: 512×512 pixels
- **Only 5 pathologies**: Cardiomegaly, Edema, Consolidation, Atelectasis, Effusion
- Source: JF Healthcare baseline model
- Use case: Focused on 5 most common/critical findings

### 4. MobileNetV3 Classifier (Custom Trainable)

**Custom Trainable Model:**
- Architecture: MobileNetV3-Large
- Input: 224×224 pixels
- **14 pathologies**: NIH Chest X-ray14 classes
- Source: Custom training on NIH dataset
- Use case: Lightweight model, mobile deployment, custom training
- Training script: `src/train_mobilenet.py`
- Inference: ~10-20ms per image (GPU), ~5.5M parameters

### Quick Pathology Reference

**All 18 Pathologies** (densenet121-res224-all):
Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass, Hernia, Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum

**NIH/PadChest (14 pathologies):**
Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass, Hernia

**CheXpert/MIMIC (10 pathologies):**
Atelectasis, Consolidation, Pneumothorax, Edema, Effusion, Pneumonia, Cardiomegaly, Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum

**RSNA (2 pathologies):**
Pneumonia, Lung Opacity

**JF Healthcare Baseline (5 pathologies):**
Cardiomegaly, Edema, Consolidation, Atelectasis, Effusion

**MobileNetV3 Custom (14 pathologies - NIH):**
Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass, Hernia

For detailed pathology coverage by model, see [Weights Documentation](weights/README.md#pathology-coverage-by-model)

### Classifier Comparison

| Classifier | Input Size | Pathologies | Speed | Use Case | Weight Variants |
|------------|------------|-------------|-------|----------|-----------------|
| **DenseNet121** (Default) | 224×224 | 18 | Fast | General purpose | 7 variants |
| **ResNet50** | 512×512 | 18 | Slower (~3-4×) | High resolution | 1 variant |
| **JF Healthcare** | 512×512 | 5 | Slower (~3-4×) | Specialized/focused | 1 variant |
| **MobileNetV3** | 224×224 | 14 | Very Fast | Custom training, mobile | Custom trained |

**Which to use?**
- **Most cases**: DenseNet121 (fast, comprehensive)
- **Need detail**: ResNet50 (high resolution, all pathologies)
- **5 key findings only**: JF Healthcare (specialized baseline)
- **Custom training**: MobileNetV3 (trainable, lightweight, mobile-friendly)

## Understanding Model Outputs

Each model returns probability scores (0-1) for its supported pathologies. Higher scores indicate higher confidence in the presence of that pathology.

**Example interpretation:**
```
Pneumonia: 0.8234 → 82.34% confidence
Effusion: 0.5423 → 54.23% confidence  
Atelectasis: 0.2567 → 25.67% confidence
```

Use the `--threshold` parameter to control which findings are classified as "positive":
```bash
# Default threshold of 0.5 (50% confidence)
uv run python src/classify_densenet121.py --input xray.jpg

# Stricter threshold (fewer positives, higher confidence)
uv run python src/classify_densenet121.py --input xray.jpg --threshold 0.7

# More sensitive threshold (more positives, lower confidence)
uv run python src/classify_densenet121.py --input xray.jpg --threshold 0.3
```

**Note:** Different models support different pathologies. See the pathology coverage table above or [detailed documentation](weights/README.md#pathology-coverage-by-model).

## Project Structure

```
ensemble-cam2/
├── src/
│   ├── classify_densenet121.py           # DenseNet121 classifier (224×224, 18 pathologies)
│   ├── classify_resnet50.py              # ResNet50 classifier (512×512, 18 pathologies)
│   ├── classify_jfhealthcare.py          # JF Healthcare baseline (512×512, 5 pathologies)
│   ├── classify_mobilenetv3.py           # MobileNetV3 classifier (224×224, 14 pathologies)
│   ├── train_mobilenet.py                # MobileNetV3 training script (NIH dataset)
│   ├── cam_utils.py                      # Shared CAM utilities
│   ├── generate_cams.py                  # CAM generation CLI tool (P0-P3 outputs)
│   ├── generate_bbox.py                  # Bounding box generation with metrics
│   └── generate_bbox_consolidated.py     # Multi-model bbox consolidation
├── data/
│   └── INPUT/                            # X-ray images
├── OUTPUT/                               # Generated CAM visualizations + bboxes
│   └── {image_name}/                     # Per-image output directory
│       ├── _input_image_.png             # Copy of input X-ray
│       ├── {model}_{pathology}_P0.png    # CAM overlay
│       ├── {model}_{pathology}_P1.png    # Heatmap only
│       ├── {model}_{pathology}_P2.png    # Red channel
│       ├── {model}_{pathology}_P3.png    # Binary mask
│       ├── {model}_{pathology}_PBBox.json   # Bbox + metrics
│       ├── {model}_{pathology}_PBBox.png    # Bbox visualization
│       ├── {pathology}_consolidated.png     # Merged mask (multi-model)
│       ├── {pathology}_consolidated_bbox.json  # Consolidated metrics
│       ├── {pathology}_consolidated_bbox.png   # Consolidated visualization
│       └── metadata.json                 # Classification results
├── weights/
│   ├── models_data/                      # Pre-trained model weights cache
│   ├── train_weight_mobilenet/           # MobileNetV3 training artifacts
│   │   ├── mobilenetv3_best.pt           # Best trained model checkpoint
│   │   ├── mobilenetv3_last.pt           # Last checkpoint
│   │   ├── performance.csv               # Training metrics log
│   │   └── README.md                     # Training documentation
│   └── README.md                         # Weights documentation
├── docs/
│   ├── about torchxrayvision.md          # Library overview
│   ├── about pytorchgradcam.md           # Grad-CAM documentation
│   ├── setup.md                          # Setup guide
│   └── model-comparison.md               # Model comparison guide
├── app.py                                # Streamlit web interface
├── config_classifiers.toml               # Classifier configuration
├── pyproject.toml                        # Project configuration
├── uv.lock                               # Dependency lock file
└── README.md                             # This file
```

## Development

### Adding Dependencies

```bash
uv add package-name
```

### Running Python Scripts

```bash
uv run python script.py
```

### Activating Virtual Environment

```bash
source .venv/bin/activate
```

## References

- [TorchXRayVision GitHub](https://github.com/mlmed/torchxrayvision)
- [Research Paper](https://arxiv.org/abs/2111.00595)
- [UV Documentation](https://docs.astral.sh/uv/)


## Credits

This project builds upon several excellent open-source libraries and pre-trained models:

### Core Libraries

- **[TorchXRayVision](https://github.com/mlmed/torchxrayvision)** - Essential chest X-ray analysis library providing pre-trained models and datasets
  - Citation: Cohen et al., "TorchXRayVision: A library of chest X-ray datasets and models", Medical Imaging with Deep Learning (2022)
  - License: Apache-2.0
  
- **[PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)** - Class Activation Map visualization for visual explainability
  - Enables generation of heatmaps showing which regions influenced model predictions
  - License: MIT

- **[PyTorch](https://pytorch.org/)** - Deep learning framework powering all model inference
  - License: BSD-3-Clause

### Additional Libraries

- **[Streamlit](https://streamlit.io/)** - Interactive web UI framework
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Pillow](https://python-pillow.org/)** - Image processing
- **[OpenCV](https://opencv.org/)** - Computer vision utilities
- **[scikit-image](https://scikit-image.org/)** - Image processing algorithms
- **[Matplotlib](https://matplotlib.org/)** - Visualization

### Pre-trained Models

This project uses pre-trained chest X-ray classification models from:
- NIH ChestX-ray14 Dataset
- PadChest Dataset
- CheXpert Dataset (Stanford)
- MIMIC-CXR Dataset
- RSNA Pneumonia Detection Challenge
- JF Healthcare Baseline Models

All models are accessed through the TorchXRayVision library.

## Disclaimer

This tool is for research and educational purposes only. It should not be used for clinical diagnosis without proper validation and regulatory approval.

## License

This project uses TorchXRayVision which is licensed under Apache-2.0.
