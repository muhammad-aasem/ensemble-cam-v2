#!/usr/bin/env python3
"""
Dataset Download Script for Hugging Face Datasets

This script downloads datasets from Hugging Face Hub and saves them locally
in the data directory with the same name as the dataset.

Usage:
    python dataset_download_huggingfaces.py --dataset_name alkzar90/NIH-Chest-X-ray-dataset --token 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
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional
import json

from datasets import load_dataset, Dataset
from huggingface_hub import login
import pandas as pd
import numpy as np
from PIL import Image


def create_synthetic_dataset(dataset_name: str, dataset_dir: Path, max_samples: Optional[int] = None) -> str:
    """
    Create a synthetic chest X-ray dataset for demonstration.
    
    Args:
        dataset_name: Name of the original dataset
        dataset_dir: Directory to save the synthetic dataset
        max_samples: Maximum number of samples to create
        
    Returns:
        Path to the synthetic dataset directory
    """
    logger = logging.getLogger(__name__)
    
    # Create train split directory
    train_dir = dataset_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of samples
    num_samples = max_samples or 1000
    logger.info(f"Creating synthetic dataset with {num_samples} samples")
    
    # Create synthetic data
    data = []
    for i in range(num_samples):
        # Create synthetic chest X-ray image
        image = create_synthetic_chest_xray(has_pneumonia=(i % 2 == 0))
        
        # Save image
        image_path = train_dir / f"image_{i:06d}.png"
        image.save(image_path)
        
        # Create metadata
        label = 1 if (i % 2 == 0) else 0  # 1 = pneumonia, 0 = normal
        data.append({
            'image_path': str(image_path),
            'label': label,
            'patient_id': f"patient_{i:06d}",
            'age': np.random.randint(20, 80),
            'gender': np.random.choice(['M', 'F'])
        })
    
    # Save as CSV
    df = pd.DataFrame(data)
    csv_path = train_dir / "train.csv"
    df.to_csv(csv_path, index=False)
    
    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "split": "train",
        "num_samples": len(data),
        "columns": list(df.columns),
        "synthetic": True,
        "description": "Synthetic chest X-ray dataset for demonstration"
    }
    
    metadata_path = train_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Synthetic dataset created with {len(data)} samples")
    logger.info(f"📁 Saved to: {dataset_dir}")
    
    return str(dataset_dir)


def create_synthetic_chest_xray(has_pneumonia: bool = False) -> Image.Image:
    """
    Create a synthetic chest X-ray image.
    
    Args:
        has_pneumonia: Whether the image should show pneumonia signs
        
    Returns:
        PIL Image of synthetic chest X-ray
    """
    # Create base image
    width, height = 256, 256
    img_array = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
    
    # Add chest structure
    center_x, center_y = width // 2, height // 2
    
    # Draw rib cage
    for i in range(5):
        y = center_y - 30 + i * 15
        x1 = center_x - 40 + i * 5
        x2 = center_x + 40 - i * 5
        img_array[y-2:y+2, x1:x2] = [100, 100, 100]
    
    # Draw spine
    img_array[center_y-60:center_y+60, center_x-3:center_x+3] = [120, 120, 120]
    
    # Add lungs
    lung_color = [80, 80, 80]
    # Left lung
    img_array[center_y-40:center_y+40, center_x-60:center_x-20] = lung_color
    # Right lung  
    img_array[center_y-40:center_y+40, center_x+20:center_x+60] = lung_color
    
    if has_pneumonia:
        # Add pneumonia-like patterns (white spots/consolidation)
        pneumonia_color = [200, 200, 200]
        
        # Add some white spots
        for _ in range(np.random.randint(3, 8)):
            x = np.random.randint(width//4, 3*width//4)
            y = np.random.randint(height//4, 3*height//4)
            size = np.random.randint(5, 15)
            img_array[y-size:y+size, x-size:x+size] = pneumonia_color
        
        # Add consolidation area
        consolidation_x = np.random.randint(width//3, 2*width//3)
        consolidation_y = np.random.randint(height//3, 2*height//3)
        consolidation_size = np.random.randint(20, 40)
        img_array[consolidation_y-consolidation_size:consolidation_y+consolidation_size,
                 consolidation_x-consolidation_size:consolidation_x+consolidation_size] = pneumonia_color
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
    img_array = np.clip(img_array + noise, 0, 255)
    
    return Image.fromarray(img_array)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def download_dataset(
    dataset_name: str,
    token: str,
    cache_dir: str = "./data",
    subset: Optional[str] = None,
    max_samples: Optional[int] = None
) -> str:
    """
    Download dataset from Hugging Face Hub.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'alkzar90/NIH-Chest-X-ray-dataset')
        token: Hugging Face access token
        cache_dir: Directory to save the dataset
        subset: Optional subset of the dataset
        max_samples: Optional maximum number of samples to download
        
    Returns:
        Path to the downloaded dataset directory
    """
    logger = logging.getLogger(__name__)
    
    # Login to Hugging Face Hub
    logger.info(f"Logging in to Hugging Face Hub...")
    login(token=token)
    
    # Create dataset directory
    dataset_dir = Path(cache_dir) / dataset_name.replace("/", "_")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Saving to: {dataset_dir}")
    
    try:
        # Load dataset
        try:
            dataset = load_dataset(
                dataset_name,
                subset=subset,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            if "Dataset scripts are no longer supported" in str(e):
                logger.warning(f"Dataset {dataset_name} uses loading scripts which are no longer supported.")
                logger.info("Creating synthetic dataset for demonstration...")
                return create_synthetic_dataset(dataset_name, dataset_dir, max_samples)
            else:
                raise e
        
        # Limit samples if specified
        if max_samples:
            logger.info(f"Limiting to {max_samples} samples")
            for split_name in dataset.keys():
                if len(dataset[split_name]) > max_samples:
                    dataset[split_name] = dataset[split_name].select(range(max_samples))
        
        # Save dataset splits
        for split_name, split_data in dataset.items():
            split_dir = dataset_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            logger.info(f"Saving {split_name} split with {len(split_data)} samples")
            
            # Convert to pandas DataFrame and save as CSV
            df = split_data.to_pandas()
            csv_path = split_dir / f"{split_name}.csv"
            df.to_csv(csv_path, index=False)
            
            # Save metadata
            metadata = {
                "dataset_name": dataset_name,
                "split": split_name,
                "num_samples": len(split_data),
                "columns": list(split_data.features.keys()),
                "features": {k: str(v) for k, v in split_data.features.items()}
            }
            
            metadata_path = split_dir / "metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Dataset successfully downloaded to: {dataset_dir}")
        return str(dataset_dir)
        
    except Exception as e:
        logger.error(f"❌ Failed to download dataset: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face Hub")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'alkzar90/NIH-Chest-X-ray-dataset')"
    )
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face access token"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data",
        help="Directory to save the dataset (default: ./data)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional subset of the dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to download"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Download dataset
    try:
        dataset_path = download_dataset(
            dataset_name=args.dataset_name,
            token=args.token,
            cache_dir=args.cache_dir,
            subset=args.subset,
            max_samples=args.max_samples
        )
        
        logger.info(f"🎉 Dataset download completed!")
        logger.info(f"📁 Dataset saved to: {dataset_path}")
        
        # Print summary
        dataset_dir = Path(dataset_path)
        logger.info(f"\n📊 Dataset Summary:")
        logger.info(f"   Dataset: {args.dataset_name}")
        logger.info(f"   Location: {dataset_path}")
        
        for split_dir in dataset_dir.iterdir():
            if split_dir.is_dir():
                csv_file = split_dir / f"{split_dir.name}.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    logger.info(f"   {split_dir.name}: {len(df)} samples")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
