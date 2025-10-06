#!/usr/bin/env python3
"""
Dataset download script for NIH Chest X-ray dataset.

This script downloads the NIH Chest X-ray dataset from Hugging Face
and saves it to the specified directory structure.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import login, snapshot_download
from tqdm import tqdm
import time

# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from env_loader import setup_environment, get_huggingface_token

# Configuration
project_root = Path(__file__).parent.parent.parent
target_dir = project_root / "dataset" / "NIH-Chest-X-ray"
download_subdir = "just-downloaded"

def download_dataset():
    """Download the NIH Chest X-ray dataset from Hugging Face."""
    
    print("NIH Chest X-ray Dataset Download")
    print("=" * 50)
    
    # Setup environment and validate tokens
    if not setup_environment():
        print("Environment setup failed. Please check your .env file or environment variables.")
        return False
    
    # Check if target directory already exists
    if target_dir.exists():
        print(f"Target directory '{target_dir}' already exists. Skipping download.")
        return True
    
    try:
        # Get Hugging Face token
        hf_token = get_huggingface_token()
        if not hf_token:
            print("Error: HUGGINGFACE_HUB_TOKEN not found.")
            return False
        
        # Login to Hugging Face
        login(token=hf_token)
        print("Successfully logged in to Hugging Face.")
        
        # Create the target directory structure
        target_dir.mkdir(parents=True, exist_ok=True)
        download_path = target_dir / download_subdir
        download_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading dataset to: {download_path}")
        print("This may take a while depending on your internet connection...")
        
        # Download the dataset
        start_time = time.time()
        
        snapshot_download(
            repo_id="alkzar90/NIH-Chest-X-ray-dataset",
            repo_type="dataset",
            local_dir=download_path,
            ignore_patterns=["*.py"],  # Skip loading scripts
            token=hf_token
        )
        
        download_time = time.time() - start_time
        
        print(f"Download completed successfully in {download_time/60:.1f} minutes!")
        print(f"Dataset saved to: {download_path}")
        
        # Show directory contents
        files = list(download_path.iterdir())
        print(f"Downloaded {len(files)} files:")
        for file in sorted(files)[:10]:  # Show first 10 files
            print(f"  - {file.name}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def main():
    """Main function."""
    success = download_dataset()
    
    if success:
        print("\nNext steps:")
        print("1. Run dataset_2_extract.py to extract zip files")
        print("2. Run dataset_3_subset-bbox.py to create BBox subset")
        print("3. Run dataset_4_classification.py to create classification dataset")
        sys.exit(0)
    else:
        print("\nDownload failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
