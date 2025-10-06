# Ensemble CAM - NIH Chest X-ray Dataset

A comprehensive deep learning pipeline for chest X-ray classification using multiple CNN architectures with Grad-CAM++ interpretability. This project provides a complete workflow from dataset preparation to model training and evaluation.

## Features

- **Complete Dataset Pipeline**: Download, extract, and preprocess NIH Chest X-ray dataset
- **Multiple CNN Architectures**: ResNet50, DenseNet121, DenseNet169, InceptionResNetV2, and Xception
- **Grad-CAM++ Compatible**: All models designed for heatmap generation and interpretability
- **Automated Workflows**: Jupyter notebooks for easy dataset preparation and model training
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and performance tracking
- **UV Package Management**: Modern Python dependency management

## Quick Start

### Option 1: Import from GitHub (Recommended)

If you don't have the project yet, use the import scripts:

#### macOS/Linux
```bash
./scripts/github_import.sh
```

#### Windows (Command Prompt)
```cmd
scripts\github_import.bat
```

#### Windows (PowerShell)
```powershell
.\scripts\github_import.ps1
```

Then follow the setup instructions below.

### Option 2: Quick Start with Jupyter Notebooks

The easiest way to get started is using the provided Jupyter notebooks:

1. **Start Jupyter Lab**:
   ```bash
   uv run jupyter lab
   ```
   This opens Jupyter Lab in your browser (usually at http://localhost:8888)

2. **Open Dataset Preparation Notebook** (Run first):
   - Navigate to `notebooks/playbook1_prepare_dataset.ipynb` in the Jupyter Lab file browser
   - Click to open the notebook
   - Run all cells to prepare the dataset

3. **Open Model Training Notebook** (Run after dataset preparation):
   - Navigate to `notebooks/playbook2_train_Xception.ipynb` in the Jupyter Lab file browser
   - Click to open the notebook
   - Run all cells to train the Xception model

### Manual Workflow (Command Line)

If you prefer command-line execution:

```bash
# Step 1: Download the dataset
uv run python src/NIH-Chest-X-ray/dataset_1_download.py

# Step 2: Extract zip files
uv run python src/NIH-Chest-X-ray/dataset_2_extract.py

# Step 3: Create object detection subset
uv run python src/NIH-Chest-X-ray/dataset_3_subset-bbox.py

# Step 4: Create classification dataset
uv run python src/NIH-Chest-X-ray/dataset_4_classification.py

# Step 5: Exclude BBox images from Classification dataset
uv run python src/NIH-Chest-X-ray/dataset_6_exclude-bbox-from-classification.py

# Step 6: Resize images for consistent training
uv run python src/NIH-Chest-X-ray/dataset_7_preprocess.py

# Step 7: Create small dataset for development (optional)
uv run python src/NIH-Chest-X-ray/dataset_8_small-dataset.py

# Step 8: Train Xception model (example)
uv run python src/train/train_Xception.py dataset/NIH-Chest-X-ray/Classification_SMALL --max-epochs 50 --batch-size 32

# Step 9: Start Jupyter Lab
uv run jupyter lab
```

### Prerequisites
- Python 3.13.2 or compatible version
- [UV](https://docs.astral.sh/uv/) package manager
- Hugging Face account and access token
- Sufficient disk space (~50GB for full dataset)

### Setup Environment

1. **Clone this repository**
   ```bash
   git clone https://github.com/muhammad-aasem/ensemble-cam-v2.git
   cd ensemble-cam-v2
   ```

2. **Initialize UV environment**
   ```bash
   uv sync
   ```
   This will:
   - Create a virtual environment with Python 3.13.2
   - Install all required dependencies (PyTorch, Jupyter Lab, datasets, etc.)

3. **Configure Hugging Face token**
   ```bash
   # Option 1: Copy and edit .env file (recommended)
   cp .env.example .env
   # Edit .env and replace placeholder with your actual token
   
   # Option 2: Set environment variable
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   ```

4. **Verify installation**
   ```bash
   uv run python --version
   ```

### Download the Dataset

Run the dataset download script:

```bash
uv run python src/NIH-Chest-X-ray/dataset_1_download.py
```

### What You'll Get

After running the script, you'll have:

- **Dataset Location**: `dataset/NIH-Chest-X-ray/just-downloaded/`
- **Dataset Size**: ~45.1 GB of medical imaging data
- **Content**: 24 files containing NIH Chest X-ray images and metadata
- **Dataset Info**: Multi-class image classification dataset for chest X-ray analysis

The script will:
- Check if the dataset already exists (skips download if found)
- Create the directory structure automatically
- Download all files to the `just-downloaded` subdirectory
- Use your Hugging Face access token for authentication
- Handle the dataset's custom loading script properly

### Extract Dataset Files

After downloading, extract the zip files:

```bash
uv run python src/NIH-Chest-X-ray/dataset_2_extract.py
```

This will:
- Find all zip files in the downloaded dataset
- Extract all zip files with progress tracking
- Delete zip files after successful extraction
- Show detailed extraction summary

### Create Object Detection Subset

After downloading and extracting the dataset, create a subset for object detection training:

```bash
uv run python src/NIH-Chest-X-ray/dataset_3_subset-bbox.py
```

This will:
- Read bounding box annotations from `BBox_List_2017.csv`
- Copy only images that have bounding box annotations
- Create a filtered CSV with clean bounding box coordinates
- Save everything to `dataset/NIH-Chest-X-ray/BBox/`

### Create Classification Dataset

Create a classification dataset organized by disease classes:

```bash
uv run python src/NIH-Chest-X-ray/dataset_4_classification.py
```

This will:
- Read disease labels from `Data_Entry_2017_v2020.csv`
- Organize images into class-specific directories
- Filter classes with minimum sample requirements
- **Move images** (not copy) to save storage space
- Create a summary CSV with class statistics
- Save everything to `dataset/NIH-Chest-X-ray/Classification/`

### Exclude BBox Images from Classification Dataset

To avoid overlap between object detection and classification training sets, remove images that are already in the BBox dataset:

```bash
uv run python src/NIH-Chest-X-ray/dataset_6_exclude-bbox-from-classification.py
```

This will:
- Read all image names from `BBox_List_2017.filtered.csv`
- Remove those images from all Classification class directories
- Update the classification summary CSV with new counts
- Provide detailed statistics on removed images per class
- Ensure no overlap between BBox and Classification datasets

### Preprocess Images for Consistent Training

To ensure consistent input sizes for model training and reduce storage requirements, preprocess all images with resizing and format conversion:

```bash
uv run python src/NIH-Chest-X-ray/dataset_7_preprocess.py
```

This will:
- Resize all images in BBox dataset to 224x224 pixels
- Resize all images in Classification dataset to 224x224 pixels
- Convert PNG files to high-quality JPEG format
- Replace original files with preprocessed versions (same filenames)
- Provide detailed statistics for each dataset
- Reduce storage requirements significantly
- Extensible framework for future preprocessing features (data augmentation, normalization, etc.)

### Create Small Dataset for Development (Optional)

For development and testing purposes, create a smaller subset of the classification dataset:

```bash
uv run python src/NIH-Chest-X-ray/dataset_8_small-dataset.py
```

This will:
- Create `Classification_SMALL/` directory with a subset of images
- Use configurable sample counts per class (adjustable in script)
- Maintain the same directory structure as the full dataset
- Provide reproducible sampling with random seed
- Create detailed statistics and summary CSV
- Generate `dataset_split.csv` with train/val/test splits (70%/20%/10%)
- Generate bar charts and summary tables showing distribution
- Support both file copying and symbolic linking modes
- Reduce dataset size by ~98% for quick development
- Handle classes with insufficient samples for stratified splitting

**Configuration**: Edit `CLASS_SAMPLE_COUNTS` in the script to adjust sample sizes per class.

**Generated Files**:
- `dataset_split.csv`: Train/validation/test splits for training
- `small_dataset_summary.csv`: Detailed statistics and metadata
- `Classification_SMALL_distribution_bar_chart.png`: Bar chart visualization
- `Classification_SMALL_summary_table.png`: Summary table with statistics

### Train Models (Optional)

Train fine-tuned models for chest X-ray classification:

#### ResNet50 Model

```bash
# Train on small dataset (recommended for development)
uv run python src/train/train_resnet50.py dataset/NIH-Chest-X-ray/Classification_SMALL --max-epochs 50 --batch-size 32

# Train on full dataset (requires significant compute resources)
uv run python src/train/train_resnet50.py dataset/NIH-Chest-X-ray/Classification --max-epochs 100 --batch-size 16
```

#### DenseNet121 Model

```bash
# Train on small dataset (recommended for development)
uv run python src/train/train_DenseNet121.py dataset/NIH-Chest-X-ray/Classification_SMALL --max-epochs 50 --batch-size 32

# Train on full dataset (requires significant compute resources)
uv run python src/train/train_DenseNet121.py dataset/NIH-Chest-X-ray/Classification --max-epochs 100 --batch-size 16
```

#### DenseNet169 Model

```bash
# Train on small dataset (recommended for development)
uv run python src/train/train_DenseNet169.py dataset/NIH-Chest-X-ray/Classification_SMALL --max-epochs 50 --batch-size 32

# Train on full dataset (requires significant compute resources)
uv run python src/train/train_DenseNet169.py dataset/NIH-Chest-X-ray/Classification --max-epochs 100 --batch-size 16
```

#### InceptionResNetV2 Model

```bash
# Train on small dataset (recommended for development)
uv run python src/train/train_InceptionResNetV2.py dataset/NIH-Chest-X-ray/Classification_SMALL --max-epochs 50 --batch-size 32

# Train on full dataset (requires significant compute resources)
uv run python src/train/train_InceptionResNetV2.py dataset/NIH-Chest-X-ray/Classification --max-epochs 100 --batch-size 16
```

#### Xception Model

```bash
# Train on small dataset (recommended for development)
uv run python src/train/train_Xception.py dataset/NIH-Chest-X-ray/Classification_SMALL --max-epochs 50 --batch-size 32

# Train on full dataset (requires significant compute resources)
uv run python src/train/train_Xception.py dataset/NIH-Chest-X-ray/Classification --max-epochs 100 --batch-size 16
```

All models will:
- Use ImageNet pretrained weights (ResNet50, DenseNet121, DenseNet169, InceptionResNetV2, or Xception)
- Fine-tune for chest X-ray classification (15 classes)
- Save best weights to respective model directories
- Generate timestamped training performance CSV files
- Support checkpoint loading for resuming training
- Include detailed metrics (accuracy, F1, precision, recall)
- Generate confusion matrix and test evaluation results
- Compatible with Grad-CAM++ for heatmap generation
- Support mixed precision training (CUDA) and MPS acceleration

**Command-line Options**:
- `dataset_path`: Path to dataset directory (required)
- `--max-epochs`: Maximum number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay for regularization (default: 1e-4)
- `--early-stopping-patience`: Early stopping patience (default: 10)
- `--dropout-rate`: Dropout rate in final layer (default: 0.5)
- `--num-workers`: Data loading workers (default: 4)

**Generated Files**:

**ResNet50 Model**:
- `models/resnet50/resnet50_best.pth`: Best model weights
- `models/resnet50/resnet50_latest.pth`: Latest checkpoint
- `models/resnet50/{timestamp}_training_performance.csv`: Training metrics per epoch
- `models/resnet50/test_results.json`: Test evaluation results
- `models/resnet50/confusion_matrix.png`: Confusion matrix visualization
- `models/resnet50/class_indices.json`: Class name to index mapping
- `models/resnet50/training_history.json`: Complete training history

**DenseNet121 Model**:
- `models/densenet121/densenet121_best.pth`: Best model weights
- `models/densenet121/densenet121_latest.pth`: Latest checkpoint
- `models/densenet121/{timestamp}_training_performance.csv`: Training metrics per epoch
- `models/densenet121/test_results.json`: Test evaluation results
- `models/densenet121/confusion_matrix.png`: Confusion matrix visualization
- `models/densenet121/class_indices.json`: Class name to index mapping
- `models/densenet121/training_history.json`: Complete training history

**DenseNet169 Model**:
- `models/densenet169/densenet169_best.pth`: Best model weights
- `models/densenet169/densenet169_latest.pth`: Latest checkpoint
- `models/densenet169/{timestamp}_training_performance.csv`: Training metrics per epoch
- `models/densenet169/test_results.json`: Test evaluation results
- `models/densenet169/confusion_matrix.png`: Confusion matrix visualization
- `models/densenet169/class_indices.json`: Class name to index mapping
- `models/densenet169/training_history.json`: Complete training history

**InceptionResNetV2 Model**:
- `models/inception_resnet_v2/inception_resnet_v2_best.pth`: Best model weights
- `models/inception_resnet_v2/inception_resnet_v2_latest.pth`: Latest checkpoint
- `models/inception_resnet_v2/{timestamp}_training_performance.csv`: Training metrics per epoch
- `models/inception_resnet_v2/test_results.json`: Test evaluation results
- `models/inception_resnet_v2/confusion_matrix.png`: Confusion matrix visualization
- `models/inception_resnet_v2/class_indices.json`: Class name to index mapping
- `models/inception_resnet_v2/training_history.json`: Complete training history

**Xception Model**:
- `models/xception/xception_best.pth`: Best model weights
- `models/xception/xception_latest.pth`: Latest checkpoint
- `models/xception/{timestamp}_training_performance.csv`: Training metrics per epoch
- `models/xception/test_results.json`: Test evaluation results
- `models/xception/confusion_matrix.png`: Confusion matrix visualization
- `models/xception/class_indices.json`: Class name to index mapping
- `models/xception/training_history.json`: Complete training history

### Clean Up Downloaded Files (Optional)

After creating both datasets, you can optionally clean up the original downloaded files to save storage space:

```bash
uv run python src/NIH-Chest-X-ray/dataset_5_cleanup.py
```

This will:
- Calculate the size of the `just-downloaded` directory
- Verify that BBox and Classification datasets exist (safety check)
- Automatically delete the entire `just-downloaded` directory
- Free up significant storage space
- Leave only the processed BBox and Classification datasets

**Warning**: This permanently deletes the original downloaded files. The script will automatically verify that both datasets exist before proceeding.

### Jupyter Notebooks

The project includes two comprehensive Jupyter notebooks for easy workflow execution:

#### 1. Dataset Preparation Notebook (`playbook1_prepare_dataset.ipynb`)
- **Purpose**: Complete dataset preparation workflow
- **Steps**: Download → Extract → Create subsets → Preprocess → Cleanup
- **Output**: Ready-to-use datasets for training
- **Duration**: 2-4 hours depending on internet speed

#### 2. Model Training Notebook (`playbook2_train_Xception.ipynb`)
- **Purpose**: Train Xception model with comprehensive evaluation
- **Features**: Automatic dataset selection, progress tracking, visualization
- **Output**: Trained model, metrics, and performance analysis
- **Duration**: 1-3 hours depending on hardware

### Start Jupyter Lab

```bash
uv run jupyter lab
```

This will open Jupyter Lab in your browser (usually at http://localhost:8888) where you can:
- Navigate to the `notebooks/` directory
- Open `playbook1_prepare_dataset.ipynb` for dataset preparation
- Open `playbook2_train_Xception.ipynb` for model training
- Explore the project files and datasets

## GitHub Scripts

The project includes cross-platform scripts for GitHub operations:

### Upload to GitHub

Upload your local changes to the GitHub repository:

#### macOS/Linux
```bash
./scripts/github_upload.sh
```

#### Windows (Command Prompt)
```cmd
scripts\github_upload.bat
```

#### Windows (PowerShell)
```powershell
.\scripts\github_upload.ps1
```

### Import from GitHub

Clone the repository from GitHub:

#### macOS/Linux
```bash
./scripts/github_import.sh
```

#### Windows (Command Prompt)
```cmd
scripts\github_import.bat
```

#### Windows (PowerShell)
```powershell
.\scripts\github_import.ps1
```

**Upload Script Features:**
- Automatic environment variable loading from `.env` file
- Sensitive file cleanup (removes token references)
- Retry logic for GitHub push protection violations
- Cross-platform compatibility
- Secure token handling

**Import Script Features:**
- Automatic repository cloning
- Git installation check
- Safety warnings for existing repositories
- Clear next-step instructions

## Project Structure

```
ensemble-cam-v2/
├── notebooks/
│   ├── playbook1_prepare_dataset.ipynb               # Dataset preparation workflow
│   └── playbook2_train_Xception.ipynb                # Xception training workflow
├── src/
│   ├── NIH-Chest-X-ray/
│   │   ├── dataset_1_download.py                     # Dataset download script
│   │   ├── dataset_2_extract.py                      # Zip extraction script
│   │   ├── dataset_3_subset-bbox.py                  # BBox subset creation script
│   │   ├── dataset_4_classification.py               # Classification dataset script
│   │   ├── dataset_5_cleanup.py                      # Cleanup script
│   │   ├── dataset_6_exclude-bbox-from-classification.py # BBox exclusion script
│   │   ├── dataset_7_preprocess.py                   # Image preprocessing script
│   │   └── dataset_8_small-dataset.py                # Small dataset creation script
│   └── train/
│       ├── train_resnet50.py                         # ResNet50 model training script
│       ├── train_DenseNet121.py                      # DenseNet121 model training script
│       ├── train_DenseNet169.py                      # DenseNet169 model training script
│       ├── train_InceptionResNetV2.py                # InceptionResNetV2 model training script
│       └── train_Xception.py                         # Xception model training script
├── dataset/                                          # Dataset directory (created after running scripts)
│   └── NIH-Chest-X-ray/
│       ├── BBox/                                     # Object detection subset
│       ├── Classification/                           # Full classification dataset
│       └── Classification_SMALL/                     # Small development dataset
├── models/                                           # Model outputs (created after training)
│   ├── resnet50/                                     # ResNet50 model results
│   ├── densenet121/                                  # DenseNet121 model results
│   ├── densenet169/                                  # DenseNet169 model results
│   ├── inception_resnet_v2/                          # InceptionResNetV2 model results
│   └── xception/                                     # Xception model results
├── scripts/                                         # GitHub operation scripts
│   ├── README.md                                    # Scripts documentation
│   ├── github_upload.sh                             # Upload script (macOS/Linux)
│   ├── github_upload.bat                            # Upload script (Windows CMD)
│   ├── github_upload.ps1                            # Upload script (Windows PowerShell)
│   ├── github_import.sh                             # Import script (macOS/Linux)
│   ├── github_import.bat                            # Import script (Windows CMD)
│   └── github_import.ps1                            # Import script (Windows PowerShell)
├── pyproject.toml                                    # UV project configuration
├── uv.lock                                          # UV lock file
├── .gitignore                                       # Git ignore file
├── .env.example                                      # Environment variables template
├── LICENSE                                          # MIT License
└── README.md                                        # This file
```

## Dataset Information

- **Source**: [Hugging Face - alkzar90/NIH-Chest-X-ray-dataset](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)
- **Task**: Image Classification (Multi-class)
- **Language**: English
- **Size**: 100K<n<1M samples
- **Paper**: [arXiv:1705.02315](https://arxiv.org/abs/1705.02315)
- **Classes**: 15 disease categories including Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, No Finding, Nodule, Pleural Thickening, Pneumonia, and Pneumothorax

## Model Architectures

This project supports training and evaluation of multiple CNN architectures:

- **ResNet50**: Deep residual network with 50 layers
- **DenseNet121**: Densely connected convolutional network
- **DenseNet169**: Larger DenseNet variant with 169 layers
- **InceptionResNetV2**: Inception architecture with residual connections
- **Xception**: Extreme version of Inception with depthwise separable convolutions

All models are:
- Pre-trained on ImageNet
- Fine-tuned for chest X-ray classification
- Compatible with Grad-CAM++ for interpretability
- Optimized for medical image analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{ensemble-cam-v2,
  title={Ensemble CAM - NIH Chest X-ray Dataset},
  author={Muhammad Aasem},
  year={2025},
  url={https://github.com/muhammad-aasem/ensemble-cam-v2}
}
```

## Acknowledgments

- NIH Clinical Center for providing the chest X-ray dataset
- Hugging Face for dataset hosting and tools
- PyTorch team for the deep learning framework
- UV team for modern Python package management