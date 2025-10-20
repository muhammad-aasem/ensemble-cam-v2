# MobileNetV3 Training Guide

This guide explains how to train and use the custom MobileNetV3 model for chest X-ray classification.

## Overview

The MobileNetV3 model is a lightweight, efficient architecture that can be trained on the NIH Chest X-ray14 dataset for 14-class pathology classification. It's designed to be compatible with the existing TorchXRayVision infrastructure.

## Prerequisites

### 1. Download NIH Chest X-ray14 Dataset

Download the dataset from: https://nihcc.app.box.com/v/ChestXray-NIHCC

You need:
- All image files (112,120 images, ~45GB)
- `Data_Entry_2017.csv` metadata file

### 2. Extract Dataset

```bash
# Example directory structure
/path/to/nih/
├── images/
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ... (112,120 images)
└── Data_Entry_2017.csv
```

## Training

### Basic Training Command

```bash
uv run python src/train_mobilenet.py \
    --epochs 50 \
    --imgpath /path/to/nih/images \
    --csvpath /path/to/nih/Data_Entry_2017.csv
```

### Advanced Training Options

```bash
uv run python src/train_mobilenet.py \
    --epochs 100 \
    --imgpath /path/to/nih/images \
    --csvpath /path/to/nih/Data_Entry_2017.csv \
    --batch-size 32 \
    --lr 0.001 \
    --device cuda \
    --num-workers 8
```

**Parameters:**
- `--imgpath` (required): Path to NIH images directory
- `--csvpath` (required): Path to Data_Entry_2017.csv
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 16)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Device to train on - cuda, cpu, or auto (default: auto)
- `--num-workers`: Data loading workers (default: 4)

### Resuming Training

The script automatically resumes from the best checkpoint:

```bash
# Train for 50 more epochs
uv run python src/train_mobilenet.py \
    --epochs 50 \
    --imgpath /path/to/nih/images \
    --csvpath /path/to/nih/Data_Entry_2017.csv
```

## Training Output

All training artifacts are saved in `weights/train_weight_mobilenet/`:

### Checkpoints

1. **`mobilenetv3_best.pt`** - Best model (highest validation AUC)
   - Contains: model weights, optimizer state, epoch, best AUC, pathologies
   - Used for inference by default

2. **`mobilenetv3_last.pt`** - Last epoch checkpoint
   - For resuming training if interrupted

### Performance Log

**`performance.csv`** - Training metrics per epoch:
```csv
epoch,train_loss,train_auc,val_loss,val_auc,learning_rate,timestamp
1,0.245123,0.678234,0.234567,0.689012,0.00100000,2024-01-15 10:30:45
2,0.223456,0.701234,0.221345,0.712345,0.00100000,2024-01-15 10:45:22
...
```

## Using the Trained Model

### Basic Inference

```bash
uv run python src/classify_mobilenetv3.py --input path/to/xray.jpg
```

### Advanced Inference

```bash
uv run python src/classify_mobilenetv3.py \
    --input path/to/xray.jpg \
    --threshold 0.6 \
    --device cuda \
    --weights weights/train_weight_mobilenet/mobilenetv3_best.pt
```

**Parameters:**
- `--input` (required): Path to X-ray image
- `--threshold`: Prediction threshold (default: 0.5)
- `--device`: Device for inference (default: auto)
- `--weights`: Path to model weights (default: best checkpoint)

### Example Output

```
Loading model: weights/train_weight_mobilenet/mobilenetv3_best.pt
Device: cuda
Architecture: MobileNetV3-Large (224×224 input)
Model loaded successfully!
  Epoch: 50
  Best AUC: 0.7845
  Pathologies: 14

============================================================
X-RAY CLASSIFICATION RESULTS (MobileNetV3)
============================================================

All Predictions (sorted by probability):
------------------------------------------------------------
Cardiomegaly                  : 0.8234 (POSITIVE)
Effusion                      : 0.6789 (POSITIVE)
Infiltration                  : 0.5423 (POSITIVE)
Atelectasis                   : 0.4567 (negative)
...

Positive Findings (≥0.5):
  • Cardiomegaly: 0.8234
  • Effusion: 0.6789
  • Infiltration: 0.5423
============================================================
```

## Model Architecture

- **Base**: MobileNetV3-Large
- **Input**: 224×224 grayscale (1-channel)
- **Output**: 14 classes (multi-label)
- **Parameters**: ~5.5M
- **Input normalization**: TorchXRayVision standard

## Training Features

### Data Split
- 80% training, 20% validation
- Random seed: 42 (reproducible)

### Loss Function
- BCEWithLogitsLoss (multi-label classification)

### Optimizer
- Adam optimizer
- Learning rate scheduling (ReduceLROnPlateau)
- Factor: 0.5, Patience: 5 epochs

### Metrics
- Loss (BCE)
- AUC (macro-average across all classes)

### Progress Tracking
- Real-time progress bars (tqdm)
- Epoch summaries
- Automatic best model saving

## Pathologies (14 Classes)

1. Atelectasis
2. Consolidation
3. Infiltration
4. Pneumothorax
5. Edema
6. Emphysema
7. Fibrosis
8. Effusion
9. Pneumonia
10. Pleural_Thickening
11. Cardiomegaly
12. Nodule
13. Mass
14. Hernia

## Performance Tips

### Memory Management
- Reduce `--batch-size` if out of memory
- Use `--device cpu` for CPU-only training
- Default batch size 16 requires ~8GB GPU memory

### Speed Optimization
- Use `--device cuda` if GPU available
- Increase `--num-workers` for faster data loading
- Training speed: ~5-10 minutes/epoch (GPU), ~30-60 minutes/epoch (CPU)

### Best Practices
1. Start with fewer epochs (10-20) to validate setup
2. Monitor validation AUC in performance.csv
3. Training typically converges around 50-100 epochs
4. Use learning rate scheduling (automatic)

## Troubleshooting

### Dataset Not Found
```
Error: Image directory not found: /path/to/images
```
**Solution**: Verify paths with `ls` and use absolute paths

### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size: `--batch-size 8` or use CPU

### Import Errors
```
ModuleNotFoundError: No module named 'tqdm'
```
**Solution**: Run `uv sync` to install dependencies

## Compatibility

The trained model is fully compatible with:
- TorchXRayVision preprocessing
- Grad-CAM visualization (future)
- Multi-model ensemble pipelines
- Mobile deployment (ONNX export possible)

## Next Steps

After training:
1. Evaluate on test set
2. Generate CAMs for interpretability
3. Compare with other models (DenseNet121, ResNet50)
4. Deploy for inference
5. Fine-tune hyperparameters if needed

## References

- NIH Chest X-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
- MobileNetV3 Paper: https://arxiv.org/abs/1905.02244
- TorchXRayVision: https://github.com/mlmed/torchxrayvision

