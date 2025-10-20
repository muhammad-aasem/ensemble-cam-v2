# Git Repository Setup - Clean & Optimized

## ✅ Repository Cleaned Successfully

### Before vs After

| Metric | Before | After |
|--------|--------|-------|
| `.git` directory size | **4.9 GB** 😱 | **372 KB** ✅ |
| Files tracked | ~50+ (with weights) | 29 (code only) |
| Push size | ~35 MB | ~600 KB |

## 🎯 What's Included in Git

**Code & Documentation (29 files):**
- ✅ Python source files (`src/*.py`)
- ✅ Configuration files (`pyproject.toml`, `config_classifiers.toml`)
- ✅ Documentation (`README.md`, `TRAINING_GUIDE.md`, etc.)
- ✅ Documentation in `weights/` directories
- ✅ Dependencies (`uv.lock`)
- ✅ Quick reference (`quick_run.txt`)

## 🚫 What's Excluded from Git

### Large Model Weights (~400+ MB)
```
weights/models_data/*.pt          # TorchXRayVision models
weights/models_data/*.pth         # JF Healthcare baseline
weights/train_weight_mobilenet/*.pt   # Trained MobileNetV3
```

**Why excluded:** Users can download pre-trained models or train their own.

### Training Data (~45+ GB)
```
data/                             # All data directory
data/NIH-Chest-X-ray/            # NIH Chest X-ray14 dataset
data/INPUT/                      # User input images
data/_BBox_/                     # Bounding box annotations
```

**Why excluded:** Datasets are too large and available from official sources.

### Generated Outputs
```
OUTPUT/*/                        # CAM visualizations
weights/train_weight_mobilenet/performance.csv
```

**Why excluded:** These are generated during use.

### System Files
```
.DS_Store                        # macOS metadata
__pycache__/                     # Python cache
.venv/                          # Virtual environment
```

## 📋 Complete .gitignore

```gitignore
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# Virtual environments
.venv

# Large model weights and training outputs (EXCLUDED from git)
weights/models_data/*.pt
weights/models_data/*.pth
weights/train_weight_mobilenet/*.pt
weights/train_weight_mobilenet/performance.csv

# Keep documentation
!weights/README.md
!weights/train_weight_mobilenet/README.md

# CAM outputs (exclude generated visualizations, keep README)
OUTPUT/*/
!OUTPUT/README.md
!OUTPUT/.gitkeep

# Data directory (NIH dataset, input images - LARGE)
data/
data/*
!data/.gitkeep

# System files
.DS_Store
*.swp
*.swo
*~
```

## 🚀 Pushing to GitHub

Now you can push without the large files:

```bash
# First time setup
git remote add origin https://github.com/muhammad-aasem/ensemble-cam-v2.git

# Push (much smaller now!)
git push -u origin main
```

**Expected push size:** ~600 KB (instead of 35+ MB)

## 📦 For New Users Cloning the Repo

After cloning, users need to:

### 1. Download Model Weights (Optional)
```bash
# TorchXRayVision models download automatically on first use
uv run python src/classify_densenet121.py --input test.jpg
```

### 2. Train MobileNetV3 (If needed)
```bash
# Download NIH Chest X-ray14 dataset first
# Then train:
uv run python src/train_mobilenet.py \
    --epochs 50 \
    --imgpath /path/to/nih/images \
    --csvpath /path/to/Data_Entry_2017.csv
```

### 3. Prepare Data Directory
```bash
mkdir -p data/INPUT
# Copy your X-ray images to data/INPUT/
```

## 🔄 Git Workflow

### Daily Work
```bash
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "Your message"

# Push
git push
```

### Large Files Stay Local
- Model weights: Stay in `weights/` but not tracked
- Training data: Stay in `data/` but not tracked
- Outputs: Stay in `OUTPUT/` but not tracked

## ✨ Benefits

1. **Fast cloning** - No large files in history
2. **Small repository** - Easy to manage and share
3. **Clean history** - Only code and documentation
4. **Portable** - Users download what they need
5. **Flexible** - Train custom models locally

## 📊 Repository Structure

```
ensemble-cam2/                    # Git tracked ✅
├── src/                         # All Python code ✅
├── docs/                        # Documentation ✅
├── weights/                     # Only README tracked ✅
│   ├── models_data/            # ❌ Not tracked (400MB)
│   └── train_weight_mobilenet/ # ❌ Not tracked (98MB)
├── data/                        # ❌ Not tracked (45GB+)
├── OUTPUT/                      # ❌ Not tracked (generated)
├── .venv/                       # ❌ Not tracked (virtual env)
├── pyproject.toml              # ✅ Tracked
├── uv.lock                     # ✅ Tracked
└── README.md                   # ✅ Tracked
```

## 🎉 Summary

Your repository is now **clean and optimized**:
- ✅ No large files in git
- ✅ Fast push/pull operations
- ✅ Professional git hygiene
- ✅ Users can download what they need
- ✅ ~600KB instead of 35+ MB pushes

Ready to push to GitHub! 🚀

