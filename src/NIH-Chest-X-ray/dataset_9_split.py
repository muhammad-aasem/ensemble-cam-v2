#!/usr/bin/env python3
"""
Dataset Split Script for NIH Chest X-ray Classification Dataset

This script creates train/validation/test splits for the Classification dataset
with a 70/25/5 ratio and generates the required dataset_split.csv file.

The script:
1. Scans all class directories in the Classification dataset
2. Collects image paths and labels for each class
3. Performs stratified splitting (70% train, 25% validation, 5% test)
4. Creates dataset_split.csv with image paths, labels, and split assignments
5. Provides detailed statistics about the splits

Usage:
    uv run python src/NIH-Chest-X-ray/dataset_9_split.py

Output:
    - dataset/NIH-Chest-X-ray/Classification/dataset_split.csv
    - Console output with split statistics
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import random

# Configuration
project_root = Path(__file__).parent.parent.parent
CLASSIFICATION_DIR = project_root / "dataset" / "NIH-Chest-X-ray" / "Classification"
OUTPUT_CSV = CLASSIFICATION_DIR / "dataset_split.csv"

# Class names (NIH Chest X-ray dataset)
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'No_Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

# Split ratios
TRAIN_RATIO = 0.70    # 70% for training
VAL_RATIO = 0.25      # 25% for validation
TEST_RATIO = 0.05     # 5% for testing

# Random seed for reproducibility
RANDOM_SEED = 42

def collect_dataset_info():
    """Collect image paths and labels from all class directories."""
    print("Collecting dataset information...")
    
    image_paths = []
    labels = []
    class_to_idx = {}
    
    # Create class to index mapping
    for idx, class_name in enumerate(CLASS_NAMES):
        class_to_idx[class_name] = idx
    
    # Scan each class directory
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = CLASSIFICATION_DIR / class_name
        
        if not class_dir.exists():
            print(f"Warning: Class directory {class_name} not found, skipping...")
            continue
        
        # Find all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(class_dir.glob(ext))
        
        if not image_files:
            print(f"Warning: No images found in {class_name} directory")
            continue
        
        # Add to lists
        for image_path in image_files:
            image_paths.append(str(image_path))
            labels.append(class_idx)
        
        print(f"  {class_name}: {len(image_files)} images")
    
    print(f"\nTotal images collected: {len(image_paths)}")
    return image_paths, labels, class_to_idx

def create_stratified_split(image_paths, labels):
    """Create stratified train/val/test split."""
    print(f"\nCreating stratified split with ratios:")
    print(f"  Train: {TRAIN_RATIO*100:.0f}%")
    print(f"  Validation: {VAL_RATIO*100:.0f}%")
    print(f"  Test: {TEST_RATIO*100:.0f}%")
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Create DataFrame for easier handling
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels,
        'class_name': [CLASS_NAMES[label] for label in labels]
    })
    
    # First split: separate train from (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(VAL_RATIO + TEST_RATIO), 
        stratify=df['label'], 
        random_state=RANDOM_SEED
    )
    
    # Second split: separate val from test
    # Calculate the proportion of test in the temp set
    test_proportion = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=test_proportion, 
        stratify=temp_df['label'], 
        random_state=RANDOM_SEED
    )
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine all splits
    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    return split_df, train_df, val_df, test_df

def print_split_statistics(train_df, val_df, test_df):
    """Print detailed statistics about the splits."""
    print(f"\nSplit Statistics:")
    print("=" * 60)
    
    # Overall counts
    print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"Train samples: {len(train_df)} ({len(train_df)/(len(train_df) + len(val_df) + len(test_df))*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/(len(train_df) + len(val_df) + len(test_df))*100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df)/(len(train_df) + len(val_df) + len(test_df))*100:.1f}%)")
    
    print(f"\nPer-class distribution:")
    print("-" * 60)
    print(f"{'Class':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 60)
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        train_count = len(train_df[train_df['label'] == class_idx])
        val_count = len(val_df[val_df['label'] == class_idx])
        test_count = len(test_df[test_df['label'] == class_idx])
        total_count = train_count + val_count + test_count
        
        if total_count > 0:  # Only show classes that have samples
            print(f"{class_name:<20} {train_count:<8} {val_count:<8} {test_count:<8} {total_count:<8}")
    
    # Check for classes with very few samples
    print(f"\nClasses with < 10 total samples:")
    small_classes = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        total_count = len(train_df[train_df['label'] == class_idx]) + \
                     len(val_df[val_df['label'] == class_idx]) + \
                     len(test_df[test_df['label'] == class_idx])
        if 0 < total_count < 10:
            small_classes.append((class_name, total_count))
    
    if small_classes:
        for class_name, count in small_classes:
            print(f"  {class_name}: {count} samples")
        print("  Note: These classes may have issues with stratified splitting.")
    else:
        print("  None - all classes have sufficient samples for proper splitting.")

def save_split_csv(split_df):
    """Save the split information to CSV file."""
    print(f"\nSaving split information to: {OUTPUT_CSV}")
    
    # Ensure the directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    split_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Successfully saved {len(split_df)} samples to dataset_split.csv")
    
    # Verify the file was created
    if OUTPUT_CSV.exists():
        file_size = OUTPUT_CSV.stat().st_size
        print(f"File size: {file_size:,} bytes")
    else:
        print("Error: CSV file was not created successfully!")

def validate_split(split_df):
    """Validate the split to ensure it's correct."""
    print(f"\nValidating split...")
    
    # Check that all splits are present
    splits = split_df['split'].unique()
    expected_splits = {'train', 'val', 'test'}
    
    if set(splits) != expected_splits:
        print(f"Error: Expected splits {expected_splits}, got {set(splits)}")
        return False
    
    # Check that all classes are represented in each split
    for split_name in expected_splits:
        split_data = split_df[split_df['split'] == split_name]
        classes_in_split = set(split_data['label'].unique())
        expected_classes = set(range(len(CLASS_NAMES)))
        
        missing_classes = expected_classes - classes_in_split
        if missing_classes:
            missing_class_names = [CLASS_NAMES[idx] for idx in missing_classes]
            print(f"Warning: Split '{split_name}' is missing classes: {missing_class_names}")
    
    # Check for duplicate image paths
    duplicates = split_df['image_path'].duplicated()
    if duplicates.any():
        print(f"Warning: Found {duplicates.sum()} duplicate image paths")
        return False
    
    print("Split validation passed!")
    return True

def main():
    """Main function to create dataset split."""
    print("NIH Chest X-ray Dataset Split Creation")
    print("=" * 50)
    
    # Check if Classification directory exists
    if not CLASSIFICATION_DIR.exists():
        print(f"Error: Classification directory not found: {CLASSIFICATION_DIR}")
        print("Please run the dataset preparation scripts first.")
        return False
    
    # Check if there are any class directories
    class_dirs = [d for d in CLASSIFICATION_DIR.iterdir() if d.is_dir() and d.name in CLASS_NAMES]
    if not class_dirs:
        print(f"Error: No class directories found in {CLASSIFICATION_DIR}")
        print("Please ensure the Classification dataset has been created.")
        return False
    
    print(f"Found {len(class_dirs)} class directories")
    
    try:
        # Collect dataset information
        image_paths, labels, class_to_idx = collect_dataset_info()
        
        if not image_paths:
            print("Error: No images found in the dataset")
            return False
        
        # Create stratified split
        split_df, train_df, val_df, test_df = create_stratified_split(image_paths, labels)
        
        # Print statistics
        print_split_statistics(train_df, val_df, test_df)
        
        # Validate split
        if not validate_split(split_df):
            print("Error: Split validation failed")
            return False
        
        # Save to CSV
        save_split_csv(split_df)
        
        print(f"\nDataset split created successfully!")
        print(f"Output file: {OUTPUT_CSV}")
        print(f"Split ratios: {TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% validation, {TEST_RATIO*100:.0f}% test")
        
        return True
        
    except Exception as e:
        print(f"Error creating dataset split: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
