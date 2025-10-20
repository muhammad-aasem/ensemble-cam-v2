# MobileNetV3 Training Directory

This directory stores all training artifacts for the MobileNetV3 model trained on NIH Chest X-ray14 dataset.

## Contents

- `mobilenetv3_best.pt` - Best model checkpoint (highest validation AUC)
- `mobilenetv3_last.pt` - Last model checkpoint (most recent epoch)
- `performance.csv` - Training and validation metrics per epoch

## Training

Train the model using:

```bash
uv run python src/train_mobilenet.py --epochs 50 --batch-size 16 --lr 0.001
```

## Checkpoint Format

Each checkpoint contains:
- `model_state_dict` - Model weights
- `optimizer_state_dict` - Optimizer state for resuming
- `epoch` - Training epoch number
- `best_auc` - Best validation AUC achieved
- `pathologies` - List of pathology labels (14 NIH classes)

## Performance CSV

The `performance.csv` file logs:
- `epoch` - Epoch number
- `train_loss` - Training loss
- `train_auc` - Training AUC (macro-average)
- `val_loss` - Validation loss
- `val_auc` - Validation AUC (macro-average)
- `learning_rate` - Current learning rate
- `timestamp` - Date and time of epoch completion

## Resuming Training

The training script automatically resumes from the best checkpoint if it exists:

```bash
# Train for 50 more epochs
uv run python src/train_mobilenet.py --epochs 50
```

## Using the Trained Model

After training, use the classifier:

```bash
uv run python src/classify_mobilenetv3.py --input path/to/xray.jpg
```

## Model Architecture

- **Base**: MobileNetV3-Large
- **Input**: 224×224 grayscale images
- **Output**: 14 pathology predictions (multi-label)
- **Parameters**: ~5.5M
- **Speed**: ~10-20ms per image (GPU)

## Pathologies (NIH Chest X-ray14)

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

