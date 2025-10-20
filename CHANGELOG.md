# Changelog

## [Unreleased] - Ensemble-CAM Web UI

### Added

#### Web UI (Major Feature)
- **`app.py`**: Streamlit-based web interface "Ensemble-CAM" for multi-model X-ray classification
  - 4-column layout: Input, Classifiers, Class Inference, Output
  - Visual image selection from `data/INPUT/` directory
  - Multi-classifier support with simultaneous execution
  - Top-3 predictions display for each classifier
  - Progress indicators and confidence visualization
  - Session state management for results persistence
  - Model caching for performance optimization

#### Configuration System
- **`config_classifiers.toml`**: Centralized classifier configuration
  - Defines available classifiers (DenseNet121, ResNet50, JF Healthcare)
  - Specifies model weights and variants
  - Input size and pathology count metadata
  - Extensible format for adding new classifiers

#### Documentation
- **`QUICKSTART.md`**: 30-second quick start guide
  - Web UI launch instructions
  - Basic CLI usage examples
  - Configuration tips
  - Troubleshooting section

- **`docs/WEB_UI_GUIDE.md`**: Comprehensive web UI documentation
  - Architecture overview
  - UI layout explanation
  - Configuration guide
  - Extension guidelines
  - Performance considerations
  - Troubleshooting guide

#### Dependencies
- Added `streamlit>=1.40.0` for web UI framework
- Added `tomli>=2.0.0` for TOML configuration parsing
- Added `scikit-image>=0.24.0` for image processing
- Updated `pillow` constraint to `>=7.1.0,<12.0.0` for Streamlit compatibility

### Changed

#### Documentation Updates
- **`README.md`**:
  - Added Web UI section as recommended usage method
  - Reorganized Usage section (Web UI first, then CLI)
  - Updated project structure with new files
  - Added links to new documentation

- **`docs/setup.md`**:
  - Added Web UI (Ensemble-CAM) section
  - Updated dependency list with new packages
  - Changed CLI section to "Running the Classifiers (CLI)"

#### Dependencies
- **`pyproject.toml`**:
  - Downgraded Pillow to `<12.0.0` for Streamlit compatibility
  - Added 3 new dependencies

### Technical Details

#### Web UI Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Ensemble-CAM                         │
├─────────────┬─────────────┬─────────────┬──────────────┤
│   Input     │ Classifiers │  Inference  │   Output     │
│             │             │             │              │
│ • Image     │ • DenseNet  │ • Results   │ • CAM        │
│   selector  │ • ResNet    │   tables    │   (future)   │
│ • Preview   │ • JFHealth  │ • Top-3     │              │
│ • Metadata  │ • Weights   │ • Scores    │              │
└─────────────┴─────────────┴─────────────┴──────────────┘
```

#### Configuration Schema
```toml
[classifiers.{classifier_id}]
name = "Display Name"
description = "Brief description"
model_type = "densenet|resnet|jfhealthcare"
input_size = 224 | 512
default_weight = "weight-name" | null
weights = ["list", "of", "weights"]
num_pathologies = 5 | 18
```

#### Caching Strategy
- **Configuration**: `@st.cache_data` - Loaded once
- **Models**: `@st.cache_resource` - Persisted across runs
- **Results**: Session state - Preserved during interaction

### Performance

#### Memory Requirements
- DenseNet121: ~500MB RAM
- ResNet50: ~2GB RAM
- JF Healthcare: ~2GB RAM
- All 3 models: ~4.5GB RAM

#### Inference Speed (CPU)
- DenseNet121: 0.5-1s per image
- ResNet50: 2-4s per image
- JF Healthcare: 2-4s per image

#### Inference Speed (GPU)
- All models: 0.1-0.5s per image

### Future Enhancements

Planned for future releases:
- [ ] CAM/Grad-CAM heatmap visualization in Output panel
- [ ] Batch image processing
- [ ] Results export (CSV/JSON)
- [ ] Adjustable confidence thresholds in UI
- [ ] Ensemble prediction aggregation
- [ ] Model comparison tables
- [ ] Pathology filtering
- [ ] Uncertainty visualization

### Migration Guide

#### For Existing Users

No breaking changes. All existing CLI commands continue to work:

```bash
# Previous usage (still works)
uv run python src/classify_densenet121.py --input xray.jpg

# New web UI (recommended)
uv run streamlit run app.py
```

#### Setup Updates

1. Update dependencies:
```bash
uv sync
```

2. Add images to INPUT directory:
```bash
mkdir -p data/INPUT
cp your-xrays/*.jpg data/INPUT/
```

3. Launch web UI:
```bash
uv run streamlit run app.py
```

### Known Issues

None at this time.

### Breaking Changes

None. All existing functionality preserved.

---

## Version History

### [0.1.0] - Initial Release
- DenseNet121 classifier with multiple weight variants
- ResNet50 high-resolution classifier
- JF Healthcare baseline classifier
- Local weight storage system
- Comprehensive documentation
- UV-based dependency management

