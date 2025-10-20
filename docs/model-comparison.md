# Model Weight Comparison Guide

This guide helps you choose the right model weight for your specific use case.

## Quick Decision Tree

```
Need comprehensive pathology detection?
├─ YES → Use densenet121-res224-all (18 pathologies) ⭐ RECOMMENDED
└─ NO → Continue below

Working with a specific dataset?
├─ NIH ChestX-ray14 → Use densenet121-res224-nih (14 pathologies)
├─ PadChest → Use densenet121-res224-pc (14 pathologies)
├─ CheXpert → Use densenet121-res224-chex (10 pathologies)
├─ MIMIC-CXR → Use densenet121-res224-mimic_nb or mimic_ch (10 pathologies)
└─ RSNA Pneumonia → Use densenet121-res224-rsna (2 pathologies)

Only need pneumonia detection?
└─ YES → Use densenet121-res224-rsna (fastest, most focused)
```

## Detailed Comparison

### 🏆 densenet121-res224-all (RECOMMENDED)

**Best for:** General-purpose X-ray classification

**Pathologies (18 total):**
✅ All pathologies from all other models combined

**Advantages:**
- Most comprehensive coverage
- Best for unknown or varied X-ray types
- Trained on the largest dataset combination
- Most robust across different imaging conditions

**Example Command:**
```bash
uv run python src/classify_densenet121.py --input xray.jpg
```

**Use When:**
- You're not sure what pathologies might be present
- You need broad screening capability
- You want the most versatile model
- You're processing X-rays from various sources

---

### 🏥 densenet121-res224-nih

**Best for:** NIH ChestX-ray14 dataset compatibility

**Pathologies (14 total):**
Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass, Hernia

**Missing:** Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum

**Example Command:**
```bash
uv run python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-nih
```

**Use When:**
- Working with NIH dataset for research
- Need benchmark comparison with NIH studies
- Want classic 14 pathology detection

---

### 🇪🇺 densenet121-res224-pc

**Best for:** PadChest dataset (European dataset)

**Pathologies (14 total):**
Same as NIH but with Fracture instead of empty slots

**Example Command:**
```bash
uv run python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-pc
```

**Use When:**
- Working with European X-ray data
- Need PadChest benchmark compatibility
- Similar to NIH but trained on European population

---

### 🎓 densenet121-res224-chex

**Best for:** Stanford CheXpert dataset

**Pathologies (10 total):**
Atelectasis, Consolidation, Pneumothorax, Edema, Effusion, Pneumonia, Cardiomegaly, Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum

**Missing:** Infiltration, Emphysema, Fibrosis, Pleural Thickening, Nodule, Mass, Hernia

**Focus:** Key clinical findings, modern pathology set

**Example Command:**
```bash
uv run python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-chex
```

**Use When:**
- Working with CheXpert dataset
- Need Stanford benchmark compatibility
- Focus on most clinically relevant findings

---

### 🏨 densenet121-res224-mimic_nb / mimic_ch

**Best for:** MIMIC-CXR dataset (US hospital data)

**Pathologies (10 total):**
Same as CheXpert (Atelectasis, Consolidation, Pneumothorax, Edema, Effusion, Pneumonia, Cardiomegaly, Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum)

**Difference:**
- `mimic_nb`: Frontal views (PA/AP)
- `mimic_ch`: All chest views

**Example Command:**
```bash
# For frontal views
uv run python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-mimic_nb

# For any chest view
uv run python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-mimic_ch
```

**Use When:**
- Working with MIMIC-CXR dataset
- Processing US hospital X-rays
- Need ICU-specific training data

---

### 🫁 densenet121-res224-rsna

**Best for:** Pneumonia detection ONLY

**Pathologies (2 total):**
- Pneumonia
- Lung Opacity

**Example Command:**
```bash
uv run python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-rsna
```

**Use When:**
- You ONLY care about pneumonia
- Want fastest inference (only 2 outputs)
- Working with RSNA Pneumonia Challenge data
- Doing pneumonia screening at scale

**⚠️ Warning:** This model will NOT detect other pathologies!

---

## Performance Comparison

| Model | Pathologies | Inference Speed | Use Case | Recommended |
|-------|-------------|-----------------|----------|-------------|
| **all** | 18 | Normal | General purpose | ⭐⭐⭐⭐⭐ |
| **nih** | 14 | Normal | NIH dataset | ⭐⭐⭐⭐ |
| **pc** | 14 | Normal | PadChest dataset | ⭐⭐⭐⭐ |
| **chex** | 10 | Normal | CheXpert dataset | ⭐⭐⭐ |
| **mimic_nb/ch** | 10 | Normal | MIMIC dataset | ⭐⭐⭐ |
| **rsna** | 2 | Fastest | Pneumonia only | ⭐⭐ (specialized) |

*Note: All models have similar inference speed since they use the same DenseNet121 architecture. The difference is negligible.*

## Real-World Examples

### Example 1: Hospital Screening

**Scenario:** You're implementing an X-ray screening tool for a hospital

**Recommendation:** `densenet121-res224-all`

**Reasoning:** Need comprehensive coverage for varied pathologies

```bash
uv run python src/classify_densenet121.py --input patient_xray.jpg --threshold 0.6
```

---

### Example 2: Pneumonia-Specific Screening

**Scenario:** COVID-19 pandemic pneumonia screening

**Recommendation:** `densenet121-res224-rsna`

**Reasoning:** Fast, focused detection of pneumonia only

```bash
uv run python src/classify_densenet121.py \
    --input patient_xray.jpg \
    --weights densenet121-res224-rsna \
    --threshold 0.5
```

---

### Example 3: Research Benchmark

**Scenario:** Publishing research paper comparing with NIH benchmarks

**Recommendation:** `densenet121-res224-nih`

**Reasoning:** Need exact NIH pathology set for fair comparison

```bash
uv run python src/classify_densenet121.py \
    --input research_xray.jpg \
    --weights densenet121-res224-nih
```

---

### Example 4: Clinical Decision Support

**Scenario:** Assisting radiologists with comprehensive analysis

**Recommendation:** `densenet121-res224-all` with lower threshold

**Reasoning:** Don't miss anything - high sensitivity

```bash
uv run python src/classify_densenet121.py \
    --input clinical_xray.jpg \
    --weights densenet121-res224-all \
    --threshold 0.3
```

---

## Summary Table

| Question | Answer | Recommended Weight |
|----------|--------|-------------------|
| Need maximum pathology coverage? | Yes | `densenet121-res224-all` ⭐ |
| Working with NIH dataset? | Yes | `densenet121-res224-nih` |
| Working with CheXpert? | Yes | `densenet121-res224-chex` |
| Working with MIMIC? | Yes | `densenet121-res224-mimic_nb/ch` |
| Only need pneumonia? | Yes | `densenet121-res224-rsna` |
| Unsure what you need? | Yes | `densenet121-res224-all` ⭐ |

**When in doubt, use `densenet121-res224-all` - it's the default for a reason!**

---

## Technical Notes

### All Models Share:
- Same architecture: DenseNet121
- Same input size: 224×224 pixels
- Same preprocessing requirements
- Similar file size: ~27MB each
- Similar inference speed

### Models Differ In:
- Number of output pathologies
- Training datasets
- Pathology label alignments
- Dataset-specific biases

### Memory Requirements:
All models have similar memory footprint:
- RAM: ~2GB
- GPU VRAM (if using CUDA): ~1GB
- Model file: ~27MB

---

## Migration Guide

**Switching from one model to another:**

Simply change the `--weights` parameter:

```bash
# From default
uv run python src/classify_densenet121.py --input xray.jpg

# To NIH-specific
uv run python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-nih

# To pneumonia-only
uv run python src/classify_densenet121.py --input xray.jpg --weights densenet121-res224-rsna
```

No code changes required! The script automatically adapts to each model's pathology set.

