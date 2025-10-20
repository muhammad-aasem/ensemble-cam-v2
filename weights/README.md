# Model Weights Directory

This directory stores the pre-trained DenseNet121 model weights locally for portability.

## Current Weights

The following model weights are currently stored:

### Downloaded Models

- **`nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`** (~27MB)
  - Model: `densenet121-res224-all` (default)
  - Trained on: All datasets combined (NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI, Kaggle)
  - Use: General-purpose chest X-ray classification

- **`nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`** (~27MB)
  - Model: `densenet121-res224-nih`
  - Trained on: NIH ChestX-ray14 dataset
  - Use: NIH-specific X-ray classification

## Directory Structure

```
weights/
└── models_data/
    ├── nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
    └── nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt
```

## Automatic Downloads

When you run the classifier with a specific `--weights` parameter, if the model is not already in this directory, it will be automatically downloaded from the TorchXRayVision repository.

Example:
```bash
# This will download the RSNA-specific model if not present
uv run python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-rsna
```

## Available Model Weights

| Weight Name | File Size | Pathologies Count | Best For |
|------------|-----------|-------------------|----------|
| `densenet121-res224-all` | ~27MB | **18 pathologies** | **General use (Recommended)** |
| `densenet121-res224-nih` | ~27MB | 14 pathologies | NIH dataset |
| `densenet121-res224-pc` | ~27MB | 14 pathologies | PadChest dataset |
| `densenet121-res224-chex` | ~27MB | 10 pathologies | CheXpert dataset |
| `densenet121-res224-mimic_nb` | ~27MB | 10 pathologies | MIMIC frontal views |
| `densenet121-res224-mimic_ch` | ~27MB | 10 pathologies | MIMIC chest views |
| `densenet121-res224-rsna` | ~27MB | 2 pathologies | Pneumonia detection only |

**Total Size (all models):** ~189MB

## Pathology Coverage by Model

This table shows which pathologies each model can classify:

| Pathology | All | NIH | PC | CheX | MIMIC-NB | MIMIC-CH | RSNA |
|-----------|:---:|:---:|:--:|:----:|:--------:|:--------:|:----:|
| **Atelectasis** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Consolidation** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Infiltration** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Pneumothorax** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Edema** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Emphysema** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Fibrosis** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Effusion** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Pneumonia** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Pleural Thickening** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Cardiomegaly** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Nodule** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Mass** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Hernia** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Lung Lesion** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Fracture** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Lung Opacity** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Enlarged Cardiomediastinum** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Total Pathologies** | **18** | **14** | **14** | **10** | **10** | **10** | **2** |

### Model Selection Guide

**Choose `densenet121-res224-all` if:**
- ✅ You want the most comprehensive pathology detection (18 total)
- ✅ You're unsure which specific pathologies you need
- ✅ You want the best general-purpose model
- ✅ **This is the default and recommended model**

**Choose `densenet121-res224-nih` if:**
- You're working with NIH ChestX-ray14 dataset
- You need the classic 14 NIH pathologies
- You want compatibility with NIH benchmark results

**Choose `densenet121-res224-pc` if:**
- You're working with PadChest dataset
- You need European dataset training

**Choose `densenet121-res224-chex` if:**
- You're working with CheXpert dataset
- You need Stanford-specific training
- You want focus on 10 key clinical findings

**Choose `densenet121-res224-mimic_nb` or `mimic_ch` if:**
- You're working with MIMIC-CXR dataset
- You need models trained on US hospital data
- You want focus on common clinical findings (10 pathologies)

**Choose `densenet121-res224-rsna` if:**
- You ONLY need pneumonia detection
- You're working with RSNA Pneumonia Challenge data
- You want a specialized pneumonia classifier (2 pathologies only: Pneumonia + Lung Opacity)

## Portability

This local storage approach ensures that:
- ✅ All model weights are contained within the project directory
- ✅ The project can be copied/moved as a single package
- ✅ No dependency on `~/.torchxrayvision/` home directory
- ✅ Works offline once weights are downloaded
- ✅ Easy to version control (if desired)

## Git Considerations

You can choose to:

1. **Include weights in git** (for complete portability):
   - Remove `weights/` from `.gitignore`
   - Commit the weights directory
   - Note: This will increase repository size

2. **Exclude weights from git** (recommended):
   - Keep `weights/` in `.gitignore`
   - Users download weights on first run
   - Include download instructions in documentation

## Manual Weight Management

To manually add/remove weights:

```bash
# List current weights
ls -lh weights/models_data/

# Remove specific weight
rm weights/models_data/nih-densenet121-*.pt

# Clear all weights (they'll re-download when needed)
rm -rf weights/models_data/*.pt

# Copy weights from another location
cp /path/to/model.pt weights/models_data/
```

## Disk Usage

Check total disk space used by weights:

```bash
du -sh weights/
```

