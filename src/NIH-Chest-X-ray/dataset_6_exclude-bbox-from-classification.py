#!/usr/bin/env python3
"""
Dataset exclusion script for NIH Chest X-ray dataset.

This script removes images from the Classification dataset that are already
present in the BBox dataset to avoid overlap between object detection and
classification training sets.
"""

import csv
import os
from pathlib import Path
from tqdm import tqdm

# --- paths (edit as needed)
# Use absolute paths from project root
project_root = Path(__file__).parent.parent.parent
BBOX_CSV_PATH = project_root / "dataset" / "NIH-Chest-X-ray" / "BBox" / "BBox_List_2017.filtered.csv"
CLASSIFICATION_DIR = project_root / "dataset" / "NIH-Chest-X-ray" / "Classification"


def get_bbox_images():
    """
    Read the BBox CSV file and extract all image names.
    
    Returns:
        set: Set of image names that have bounding box annotations
    """
    bbox_images = set()
    
    if not BBOX_CSV_PATH.exists():
        print(f"Error: BBox CSV file not found at {BBOX_CSV_PATH}")
        print("Please run dataset_3_subset-bbox.py first.")
        return bbox_images
    
    print("Reading BBox annotations...")
    try:
        with open(BBOX_CSV_PATH, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row["Image Index"].strip()
                bbox_images.add(image_name)
    except Exception as e:
        print(f"Error reading BBox CSV file: {e}")
        return set()
    
    print(f"Found {len(bbox_images)} images with bounding box annotations")
    return bbox_images


def remove_bbox_images_from_classification(bbox_images):
    """
    Remove images from Classification dataset that are in the BBox dataset.
    
    Args:
        bbox_images: Set of image names to remove from classification dataset
        
    Returns:
        dict: Statistics about removed images per class
    """
    if not CLASSIFICATION_DIR.exists():
        print(f"Error: Classification directory not found at {CLASSIFICATION_DIR}")
        print("Please run dataset_4_classification.py first.")
        return {}
    
    print(f"\nScanning Classification directory: {CLASSIFICATION_DIR}")
    
    # Get all class directories
    class_dirs = [d for d in CLASSIFICATION_DIR.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print("No class directories found in Classification dataset")
        return {}
    
    print(f"Found {len(class_dirs)} class directories")
    
    removal_stats = {}
    total_removed = 0
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Get all PNG files in this class directory
        png_files = list(class_dir.glob("*.png"))
        total_images = len(png_files)
        
        if total_images == 0:
            print(f"  No PNG files found in {class_name}")
            removal_stats[class_name] = {"total": 0, "removed": 0, "remaining": 0}
            continue
        
        print(f"  Total images: {total_images}")
        
        # Find images that need to be removed
        images_to_remove = []
        for png_file in png_files:
            if png_file.name in bbox_images:
                images_to_remove.append(png_file)
        
        removed_count = len(images_to_remove)
        remaining_count = total_images - removed_count
        
        print(f"  Images to remove: {removed_count}")
        print(f"  Images remaining: {remaining_count}")
        
        # Remove the images
        if removed_count > 0:
            print(f"  Removing {removed_count} images...")
            for img_file in tqdm(images_to_remove, desc=f"Removing from {class_name}"):
                try:
                    img_file.unlink()  # Delete the file
                except Exception as e:
                    print(f"    Error removing {img_file.name}: {e}")
        
        removal_stats[class_name] = {
            "total": total_images,
            "removed": removed_count,
            "remaining": remaining_count
        }
        
        total_removed += removed_count
    
    print(f"\nTotal images removed across all classes: {total_removed}")
    return removal_stats


def update_classification_summary(removal_stats):
    """
    Update the classification summary CSV with new counts after removal.
    
    Args:
        removal_stats: Dictionary with removal statistics per class
    """
    summary_csv = CLASSIFICATION_DIR / "classification_summary.csv"
    
    if not summary_csv.exists():
        print("Warning: classification_summary.csv not found, skipping update")
        return
    
    print("\nUpdating classification summary CSV...")
    
    try:
        # Read existing summary
        with open(summary_csv, 'r', newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Update counts
        for row in rows:
            class_name = row["Class Name"]
            if class_name in removal_stats:
                stats = removal_stats[class_name]
                row["Total Images"] = str(stats["total"])
                row["Copied Images"] = str(stats["remaining"])  # Updated to remaining
                row["Missing Images"] = str(stats["removed"])  # Updated to removed
        
        # Write updated summary
        with open(summary_csv, 'w', newline='', encoding="utf-8") as f:
            fieldnames = ["Class Name", "Total Images", "Copied Images", "Missing Images"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Updated {summary_csv}")
        
    except Exception as e:
        print(f"Error updating summary CSV: {e}")


def main():
    """Main function to exclude BBox images from Classification dataset."""
    print("🔍 NIH Chest X-ray Dataset - Exclude BBox Images from Classification")
    print("=" * 70)
    
    # Step 1: Get BBox images
    bbox_images = get_bbox_images()
    if not bbox_images:
        print("No BBox images found. Exiting.")
        return False
    
    # Step 2: Remove BBox images from Classification dataset
    removal_stats = remove_bbox_images_from_classification(bbox_images)
    if not removal_stats:
        print("No classification classes found. Exiting.")
        return False
    
    # Step 3: Update summary CSV
    update_classification_summary(removal_stats)
    
    # Print final summary
    print(f"\nBBox image exclusion completed successfully!")
    print(f"Classification directory: {CLASSIFICATION_DIR}")
    print(f"BBox CSV: {BBOX_CSV_PATH}")
    
    print(f"\nSummary by class:")
    total_original = 0
    total_removed = 0
    total_remaining = 0
    
    for class_name, stats in removal_stats.items():
        total_original += stats["total"]
        total_removed += stats["removed"]
        total_remaining += stats["remaining"]
        
        print(f"  {class_name}:")
        print(f"    Original: {stats['total']:,}")
        print(f"    Removed:  {stats['removed']:,}")
        print(f"    Remaining: {stats['remaining']:,}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total original images: {total_original:,}")
    print(f"  Total removed images:  {total_removed:,}")
    print(f"  Total remaining images: {total_remaining:,}")
    print(f"  Removal rate: {(total_removed/total_original)*100:.1f}%")
    
    print(f"\nClassification dataset now excludes all BBox images!")
    print(f"   This ensures no overlap between object detection and classification training sets.")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
