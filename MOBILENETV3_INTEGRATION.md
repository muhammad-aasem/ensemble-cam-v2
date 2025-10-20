# MobileNetV3 Integration Guide

Complete integration of MobileNetV3 into the ensemble-cam2 pipeline for classification, CAM generation, and bounding box localization.

## ✅ Integration Complete

MobileNetV3 is now fully integrated into all components of the ensemble-cam2 system:

### 1. Classification ✅
- **Script:** `src/classify_mobilenetv3.py`
- **Usage:**
```bash
uv run python src/classify_mobilenetv3.py --input path/to/xray.jpg
```

### 2. CAM Generation ✅
- **Script:** `src/generate_cams.py`
- **Usage:**
```bash
# Generate CAM for specific pathology
uv run python src/generate_cams.py \
    --input path/to/xray.jpg \
    --model mobilenetv3 \
    --target-class "Cardiomegaly"

# Generate top-3 predictions
uv run python src/generate_cams.py \
    --input path/to/xray.jpg \
    --model mobilenetv3 \
    --top-k 3
```

### 3. Bounding Box Generation ✅
- **Script:** `src/generate_bbox.py`
- **Usage:**
```bash
uv run python src/generate_bbox.py \
    --input-mask OUTPUT/image_name/mobilenetv3_Cardiomegaly_P3.png \
    --ground-truth "Cardiomegaly,277,459,540,301"
```

### 4. Multi-Model Consolidation ✅
- **Script:** `src/generate_bbox_consolidated.py`
- **Usage:**
```bash
python src/generate_bbox_consolidated.py \
    --input-sources \
        OUTPUT/image/densenet121_Cardiomegaly_P3.png \
        OUTPUT/image/resnet50_Cardiomegaly_P3.png \
        OUTPUT/image/jfhealthcare_Cardiomegaly_P3.png \
        OUTPUT/image/mobilenetv3_Cardiomegaly_P3.png \
    --method merge-masking \
    --target-class "Cardiomegaly" \
    --ground-truth "Cardiomegaly,277,459,540,301"
```

## 📁 Files Modified

### Core Integration Files
1. **`src/cam_utils.py`**
   - Added MobileNetV3 model loading in `load_model_for_cam()`
   - Added target layer configuration in `get_target_layer()`
   - Handles custom checkpoint loading from `weights/train_weight_mobilenet/`

2. **`src/generate_cams.py`**
   - Added 'mobilenetv3' to model choices
   - Added model configuration for MobileNetV3
   - Input size: 224×224 (same as DenseNet121)

3. **`src/generate_bbox.py`**
   - No changes needed - already compatible with any model's P3 masks

### Documentation Updates
1. **`README.md`**
   - Added MobileNetV3 to CAM examples
   - Listed as available model

2. **`quick_run.txt`**
   - Added complete MobileNetV3 workflow
   - Added to consolidation example

## 🔄 Complete Workflow

### Step 1: Train MobileNetV3
```bash
uv run python src/train_mobilenet.py \
    --epochs 50 \
    --imgpath data/NIH-Chest-X-ray/images/images \
    --csvpath data/NIH-Chest-X-ray/Data_Entry_2017_v2020.csv \
    --batch-size 16
```

### Step 2: Classify X-ray
```bash
export input_image=data/INPUT/Cardiomegaly/00005066_030.png
uv run python src/classify_mobilenetv3.py --input $input_image
```

### Step 3: Generate CAMs
```bash
export target_class="Cardiomegaly"
uv run python src/generate_cams.py \
    --input $input_image \
    --target-class $target_class \
    --model mobilenetv3
```

**Output:**
- `OUTPUT/00005066_030/mobilenetv3_Cardiomegaly_P0.png` - Overlay
- `OUTPUT/00005066_030/mobilenetv3_Cardiomegaly_P1.png` - Heatmap
- `OUTPUT/00005066_030/mobilenetv3_Cardiomegaly_P2.png` - Red channel
- `OUTPUT/00005066_030/mobilenetv3_Cardiomegaly_P3.png` - Binary mask

### Step 4: Generate Bounding Box
```bash
uv run python src/generate_bbox.py \
    --input-mask OUTPUT/00005066_030/mobilenetv3_Cardiomegaly_P3.png \
    --ground-truth "Cardiomegaly,277,459,540,301"
```

**Output:**
- `OUTPUT/00005066_030/mobilenetv3_Cardiomegaly_PBBox.json` - Bbox + metrics
- `OUTPUT/00005066_030/mobilenetv3_Cardiomegaly_PBBox.png` - Visualization

### Step 5: Ensemble Analysis (Optional)
```bash
python src/generate_bbox_consolidated.py \
    --input-sources \
        OUTPUT/00005066_030/densenet121_Cardiomegaly_P3.png \
        OUTPUT/00005066_030/resnet50_Cardiomegaly_P3.png \
        OUTPUT/00005066_030/jfhealthcare_Cardiomegaly_P3.png \
        OUTPUT/00005066_030/mobilenetv3_Cardiomegaly_P3.png \
    --method merge-masking \
    --target-class "Cardiomegaly" \
    --ground-truth "Cardiomegaly,277,459,540,301"
```

## 🎯 Model Specifications

| Feature | Value |
|---------|-------|
| Architecture | MobileNetV3-Large |
| Input Size | 224×224 grayscale |
| Pathologies | 14 (NIH Chest X-ray14) |
| Parameters | ~5.5M |
| Speed (GPU) | ~10-20ms per image |
| Speed (CPU) | ~100-200ms per image |
| Weights | Custom trained |

## 📊 Pathologies Supported

MobileNetV3 classifies these 14 NIH pathologies:

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

## 🔧 Technical Details

### CAM Target Layer
- Uses last convolutional layer: `model.features[-1]`
- Same as DenseNet121 architecture
- Compatible with all Grad-CAM methods

### Weight Loading
- Checkpoint format: PyTorch state dict
- Location: `weights/train_weight_mobilenet/mobilenetv3_best.pt`
- Fallback: `mobilenetv3_last.pt`
- Contains: model weights, pathologies, epoch, best AUC

### Integration Architecture
```
MobileNetV3XRay
├── features (MobileNetV3 backbone)
│   └── [-1] Target layer for CAM
├── avgpool
└── classifier
    ├── Linear(960, 1280)
    ├── Hardswish
    ├── Dropout(0.2)
    └── Linear(1280, 14)
```

## ✨ Features

✅ **Full Pipeline Integration**
- Works with all CAM methods (GradCAM, GradCAM++, ScoreCAM, etc.)
- Compatible with bounding box generation
- Supports multi-model ensemble consolidation

✅ **Automatic Weight Detection**
- Loads from `weights/train_weight_mobilenet/mobilenetv3_best.pt`
- Supports custom weight paths via `--weights` parameter
- Clear error messages if weights not found

✅ **Consistent Interface**
- Same command-line interface as other models
- Same output format (P0, P1, P2, P3)
- Same naming convention for files

## 🎨 Output Examples

### Classification Output
```
============================================================
X-RAY CLASSIFICATION RESULTS (MobileNetV3)
============================================================

All Predictions (sorted by probability):
------------------------------------------------------------
Cardiomegaly                  : 0.8234 (POSITIVE)
Effusion                      : 0.6789 (POSITIVE)
Infiltration                  : 0.5423 (POSITIVE)
...
```

### CAM Output Files
```
OUTPUT/00005066_030/
├── mobilenetv3_Cardiomegaly_P0.png    # Overlay
├── mobilenetv3_Cardiomegaly_P1.png    # Heatmap
├── mobilenetv3_Cardiomegaly_P2.png    # Red channel
├── mobilenetv3_Cardiomegaly_P3.png    # Binary mask
├── mobilenetv3_Cardiomegaly_PBBox.json    # Bbox + metrics
└── mobilenetv3_Cardiomegaly_PBBox.png     # Visualization
```

## 🚀 Quick Test

Test the complete integration:

```bash
# 1. Classify
uv run python src/classify_mobilenetv3.py \
    --input data/INPUT/Cardiomegaly/00005066_030.png

# 2. Generate CAM
uv run python src/generate_cams.py \
    --input data/INPUT/Cardiomegaly/00005066_030.png \
    --model mobilenetv3 \
    --target-class "Cardiomegaly"

# 3. Check outputs
ls -la OUTPUT/00005066_030/mobilenetv3_*
```

Expected outputs:
- 4 PNG files (P0, P1, P2, P3)
- After bbox generation: 2 additional files (PBBox.json, PBBox.png)

## 📚 Documentation

For more details, see:
- **Training:** `TRAINING_GUIDE.md`
- **Training artifacts:** `weights/train_weight_mobilenet/README.md`
- **General usage:** `README.md`
- **Quick commands:** `quick_run.txt`

## ✅ Verification Checklist

- [x] Classification works with trained model
- [x] CAM generation integrated into `generate_cams.py`
- [x] All CAM outputs generated (P0, P1, P2, P3)
- [x] Bounding box generation works with MobileNetV3 masks
- [x] Multi-model consolidation includes MobileNetV3
- [x] Documentation updated
- [x] Quick reference updated
- [x] No linter errors

## 🎉 Summary

MobileNetV3 is now a **first-class citizen** in the ensemble-cam2 system:
- ✅ Same interface as DenseNet121, ResNet50, JF Healthcare
- ✅ Full CAM support with all visualization methods
- ✅ Compatible with bounding box generation
- ✅ Integrated into multi-model ensemble
- ✅ Comprehensive documentation

Your trained MobileNetV3 model can now be used in the complete pipeline from classification → CAM → bounding box → ensemble analysis!

