# Setup Guide

This guide provides multiple ways to set up the Ensemble CAM project.

## Option 1: UV Package Manager (Recommended)

UV is a modern, fast Python package manager. It's the recommended approach for this project.

### Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup Project

```bash
# Clone repository
git clone https://github.com/muhammad-aasem/ensemble-cam-v2.git
cd ensemble-cam-v2

# Install dependencies
uv sync

# Verify installation
uv run python --version
```

## Option 2: Traditional pip + venv

If you prefer using pip and virtual environments:

### Setup Project

```bash
# Clone repository
git clone https://github.com/muhammad-aasem/ensemble-cam-v2.git
cd ensemble-cam-v2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python --version
```

## Option 3: Conda

If you prefer using Conda:

### Setup Project

```bash
# Clone repository
git clone https://github.com/muhammad-aasem/ensemble-cam-v2.git
cd ensemble-cam-v2

# Create conda environment
conda create -n ensemble-cam python=3.13

# Activate environment
conda activate ensemble-cam

# Install PyTorch (choose appropriate version)
# For CPU only:
conda install pytorch torchvision -c pytorch

# For CUDA (if you have NVIDIA GPU):
# conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Verify installation
python --version
```

## Environment Configuration

The project uses a `.env` file for configuration. You need to set up your tokens:

### macOS/Linux
```bash
# Set environment variable for current session
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# Set permanently (add to shell profile)
echo 'export HUGGINGFACE_HUB_TOKEN="your_token_here"' >> ~/.zshrc  # For zsh
echo 'export HUGGINGFACE_HUB_TOKEN="your_token_here"' >> ~/.bash_profile  # For bash

# Reload shell profile
source ~/.zshrc  # or source ~/.bash_profile
```

### Windows
```cmd
# Command Prompt - Set for current session
set HUGGINGFACE_HUB_TOKEN=your_token_here

# PowerShell - Set for current session
$env:HUGGINGFACE_HUB_TOKEN="your_token_here"

# Set permanently via System Properties:
# 1. Right-click "This PC" → Properties
# 2. Advanced system settings → Environment Variables
# 3. New → Variable name: HUGGINGFACE_HUB_TOKEN
# 4. Variable value: your_token_here
# 5. OK to save
```

### Recommended: Use .env file
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env file and replace placeholder values with your actual tokens
# HUGGINGFACE_HUB_TOKEN=your_actual_token_here
# GITHUB_TOKEN=your_actual_token_here
```

The `.env` file is automatically loaded by the scripts and provides a secure way to manage your tokens.

## Verify Setup

Run this command to verify everything is working:

```bash
# With UV
uv run python -c "import torch, torchvision, datasets, jupyterlab; print('Setup successful!')"

# With pip/conda
python -c "import torch, torchvision, datasets, jupyterlab; print('Setup successful!')"
```

## Next Steps

After successful setup, you can:

1. **Start Jupyter Lab**:
   ```bash
   # With UV
   uv run jupyter lab
   
   # With pip/conda
   jupyter lab
   ```
   Then navigate to `notebooks/playbook1_prepare_dataset.ipynb` in the browser

2. **Run Individual Scripts**:
   ```bash
   # With UV
   uv run python src/NIH-Chest-X-ray/dataset_1_download.py
   
   # With pip/conda
   python src/NIH-Chest-X-ray/dataset_1_download.py
   ```

## Troubleshooting

### Common Issues

1. **Python version**: Make sure you're using Python 3.13.2 or compatible
2. **CUDA issues**: If you have GPU issues, try CPU-only PyTorch installation
3. **Memory issues**: For large datasets, ensure you have sufficient RAM (16GB+ recommended)
4. **Disk space**: The full dataset requires ~50GB of free space

### Getting Help

- Check the main [README.md](README.md) for detailed usage instructions
- Open an issue on GitHub for bugs or questions
- Review the Jupyter notebooks for step-by-step workflows
