#!/usr/bin/env python3
"""
Dataset preprocessing script for NIH Chest X-ray dataset.

This script performs various preprocessing operations on the NIH Chest X-ray dataset,
including image resizing, format conversion, and other data preparation tasks.
Currently implements image resizing functionality, with additional preprocessing
functions to be added in the future.

Current functionality:
- Resize images to standard size (224x224)
- Convert PNG to JPEG format
- Maintain consistent filenames
- Reduce storage requirements

Future functionality (to be added):
- Data augmentation
- Normalization
- Quality filtering
- Metadata extraction
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --- Configuration (edit as needed)
# Use absolute paths from project root
project_root = Path(__file__).parent.parent.parent
BBOX_DIR = project_root / "dataset" / "NIH-Chest-X-ray" / "BBox"
CLASSIFICATION_DIR = project_root / "dataset" / "NIH-Chest-X-ray" / "Classification"

# Image preprocessing parameters
TARGET_SIZE = (224, 224)  # Standard size for many deep learning models
RESIZE_MODE = Image.Resampling.LANCZOS  # High-quality resampling
JPEG_QUALITY = 95  # JPEG quality (1-100, higher = better quality)

# Future preprocessing parameters (to be implemented)
ENABLE_DATA_AUGMENTATION = False  # Will be implemented later
ENABLE_NORMALIZATION = False  # Will be implemented later
ENABLE_QUALITY_FILTERING = False  # Will be implemented later


def preprocess_image(input_path, output_path, target_size, resize_mode, quality):
    """
    Preprocess a single image (currently: resize and convert format).
    
    This function currently performs image resizing and format conversion.
    Additional preprocessing steps will be added in future versions.
    
    Args:
        input_path: Path to input image
        output_path: Path to save preprocessed image
        target_size: Tuple of (width, height) for target size
        resize_mode: PIL resampling mode
        quality: JPEG quality (1-100)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (handles RGBA, L, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Current preprocessing: Resize the image
            processed_img = img.resize(target_size, resize_mode)
            
            # Future preprocessing steps will be added here:
            # - Data augmentation (if enabled)
            # - Normalization (if enabled)
            # - Quality filtering (if enabled)
            
            # Save with high quality
            processed_img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
        return True
    except Exception as e:
        print(f"    Error preprocessing {input_path.name}: {e}")
        return False


def preprocess_images_in_directory(directory_path, target_size, resize_mode, quality):
    """
    Preprocess all PNG images in a directory (currently: resize and convert format).
    
    This function currently performs image resizing and format conversion.
    Additional preprocessing steps will be added in future versions.
    
    Args:
        directory_path: Path to directory containing images
        target_size: Tuple of (width, height) for target size
        resize_mode: PIL resampling mode
        quality: JPEG quality (1-100)
    
    Returns:
        dict: Statistics about preprocessing operation
    """
    if not directory_path.exists():
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    # Find all PNG files
    png_files = list(directory_path.glob("*.png"))
    
    if not png_files:
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    print(f"  Found {len(png_files)} PNG files")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # Process each image
    for png_file in tqdm(png_files, desc=f"Preprocessing {directory_path.name}"):
        # Create output path (same name, but as JPEG)
        output_path = png_file.with_suffix('.jpg')
        
        # Skip if output already exists
        if output_path.exists():
            skipped_count += 1
            continue
        
        # Preprocess the image (currently: resize and convert)
        if preprocess_image(png_file, output_path, target_size, resize_mode, quality):
            success_count += 1
            # Remove original PNG file after successful preprocessing
            try:
                png_file.unlink()
            except Exception as e:
                print(f"    Warning: Could not remove original {png_file.name}: {e}")
        else:
            failed_count += 1
    
    return {
        "total": len(png_files),
        "success": success_count,
        "failed": failed_count,
        "skipped": skipped_count
    }


def preprocess_bbox_dataset():
    """
    Preprocess all images in the BBox dataset (currently: resize and convert format).
    
    Returns:
        dict: Statistics about preprocessing operation
    """
    print(f"\nProcessing BBox Dataset: {BBOX_DIR}")
    
    if not BBOX_DIR.exists():
        print(f"Error: BBox directory not found at {BBOX_DIR}")
        print("Please run dataset_3_subset-bbox.py first.")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    # Preprocess images in BBox directory
    stats = preprocess_images_in_directory(BBOX_DIR, TARGET_SIZE, RESIZE_MODE, JPEG_QUALITY)
    
    print(f"  BBox preprocessing completed:")
    print(f"    Total images: {stats['total']}")
    print(f"    Successfully preprocessed: {stats['success']}")
    print(f"    Failed: {stats['failed']}")
    print(f"    Skipped: {stats['skipped']}")
    
    return stats


def preprocess_classification_dataset():
    """
    Preprocess all images in the Classification dataset (currently: resize and convert format).
    
    Returns:
        dict: Overall statistics about preprocessing operation
    """
    print(f"\nProcessing Classification Dataset: {CLASSIFICATION_DIR}")
    
    if not CLASSIFICATION_DIR.exists():
        print(f"Error: Classification directory not found at {CLASSIFICATION_DIR}")
        print("Please run dataset_4_classification.py first.")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    # Get all class directories
    class_dirs = [d for d in CLASSIFICATION_DIR.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print("No class directories found in Classification dataset")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    print(f"Found {len(class_dirs)} class directories")
    
    overall_stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    # Process each class directory
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n  Processing class: {class_name}")
        
        # Preprocess images in this class directory
        stats = preprocess_images_in_directory(class_dir, TARGET_SIZE, RESIZE_MODE, JPEG_QUALITY)
        
        # Add to overall stats
        for key in overall_stats:
            overall_stats[key] += stats[key]
        
            print(f"    {class_name}: {stats['success']}/{stats['total']} preprocessed")
    
    print(f"\n  Classification preprocessing completed:")
    print(f"    Total images: {overall_stats['total']}")
    print(f"    Successfully preprocessed: {overall_stats['success']}")
    print(f"    Failed: {overall_stats['failed']}")
    print(f"    Skipped: {overall_stats['skipped']}")
    
    return overall_stats


def main():
    """Main function to preprocess images in both datasets."""
    print("NIH Chest X-ray Dataset - Image Preprocessing")
    print("=" * 60)
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} pixels")
    print(f"Resize mode: {RESIZE_MODE}")
    print(f"JPEG quality: {JPEG_QUALITY}")
    print(f"Data augmentation: {'Enabled' if ENABLE_DATA_AUGMENTATION else 'Disabled'}")
    print(f"Normalization: {'Enabled' if ENABLE_NORMALIZATION else 'Disabled'}")
    print(f"Quality filtering: {'Enabled' if ENABLE_QUALITY_FILTERING else 'Disabled'}")
    
    # Step 1: Preprocess BBox dataset
    bbox_stats = preprocess_bbox_dataset()
    
    # Step 2: Preprocess Classification dataset
    classification_stats = preprocess_classification_dataset()
    
    # Calculate overall statistics
    total_images = bbox_stats["total"] + classification_stats["total"]
    total_success = bbox_stats["success"] + classification_stats["success"]
    total_failed = bbox_stats["failed"] + classification_stats["failed"]
    total_skipped = bbox_stats["skipped"] + classification_stats["skipped"]
    
    # Print final summary
    print(f"\nImage preprocessing completed successfully!")
    print(f"BBox directory: {BBOX_DIR}")
    print(f"Classification directory: {CLASSIFICATION_DIR}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total images processed: {total_images:,}")
    print(f"  Successfully preprocessed: {total_success:,}")
    print(f"  Failed: {total_failed:,}")
    print(f"  Skipped: {total_skipped:,}")
    
    if total_images > 0:
        success_rate = (total_success / total_images) * 100
        print(f"  Success rate: {success_rate:.1f}%")
    
    print(f"\nBBox Dataset:")
    print(f"  Images: {bbox_stats['total']:,}")
    print(f"  Preprocessed: {bbox_stats['success']:,}")
    
    print(f"\nClassification Dataset:")
    print(f"  Images: {classification_stats['total']:,}")
    print(f"  Preprocessed: {classification_stats['success']:,}")
    
    print(f"\nAll images have been preprocessed!")
    print(f"   Current preprocessing: Resize to {TARGET_SIZE[0]}x{TARGET_SIZE[1]} pixels")
    print(f"   Format conversion: PNG → JPEG (quality: {JPEG_QUALITY})")
    print(f"   This reduces storage requirements and ensures consistent input sizes for training.")
    print(f"\nFuture preprocessing features:")
    print(f"   - Data augmentation (currently disabled)")
    print(f"   - Normalization (currently disabled)")
    print(f"   - Quality filtering (currently disabled)")
    
    return total_failed == 0


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
