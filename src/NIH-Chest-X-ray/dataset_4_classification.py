#!/usr/bin/env python3
"""
Dataset classification script for NIH Chest X-ray dataset.

This script creates a classification dataset from the NIH Chest X-ray dataset
by reading the Data_Entry_2017_v2020.csv file and organizing images by their
disease labels. It creates a Classification directory with subdirectories for
each disease class and copies the corresponding images.
"""

import csv, os, shutil
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm

# --- paths (edit as needed)
# Use absolute paths from project root
project_root = Path(__file__).parent.parent.parent
CSV_PATH = project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded" / "data" / "Data_Entry_2017_v2020.csv"
# Try multiple possible image directories (after extraction)
POSSIBLE_IMG_DIRS = [
    project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded" / "data" / "images" / "images",  # Actual location after extraction
    project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded" / "data" / "images",  # Alternative location
    project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded" / "images",  # Original location
    project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded",  # Root of extracted files
]
OUT_DIR = project_root / "dataset" / "NIH-Chest-X-ray" / "Classification"  # will be created
COPY_MODE = "move"  # "copy", "symlink", or "move"
MIN_SAMPLES_PER_CLASS = 10  # Minimum samples required for a class to be included


def find_image_directory():
    """
    Find the directory containing the extracted PNG images.
    
    Returns:
        str: Path to the image directory, or None if not found.
    """
    print("Searching for image directory...")
    
    for img_dir in POSSIBLE_IMG_DIRS:
        if os.path.exists(img_dir):
            # Check if this directory contains PNG files
            png_files = list(Path(img_dir).glob("*.png"))
            if png_files:
                print(f"Found {len(png_files)} PNG files in: {img_dir}")
                return img_dir
            else:
                print(f"Directory exists but no PNG files found: {img_dir}")
    
    # If no directory found, try to find PNG files recursively
    print("Searching recursively for PNG files...")
    dataset_root = project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded"
    if dataset_root.exists():
        png_files = list(dataset_root.rglob("*.png"))
        if png_files:
            # Get the directory of the first PNG file
            img_dir = str(png_files[0].parent)
            print(f"Found {len(png_files)} PNG files in: {img_dir}")
            return img_dir
    
    return None


def parse_disease_labels(labels_str):
    """
    Parse disease labels from the CSV format.
    
    Args:
        labels_str: String containing disease labels separated by '|'
    
    Returns:
        list: List of individual disease labels
    """
    if not labels_str or labels_str.strip() == "":
        return ["No Finding"]
    
    # Split by '|' and clean up each label
    labels = [label.strip() for label in labels_str.split('|')]
    return [label for label in labels if label]  # Remove empty strings


def main():
    """Main function to create classification dataset."""
    print("Creating NIH Chest X-ray classification dataset...")
    
    # Check if CSV file exists
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        print("Please run dataset_1_download.py first to download the dataset.")
        return False
    
    # Find the image directory
    ALL_IMG_DIR = find_image_directory()
    if not ALL_IMG_DIR:
        print("Error: Could not find directory containing PNG images.")
        print("Please run dataset_2_extract.py first to extract the zip files.")
        return False
    
    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")
    print(f"Using image directory: {ALL_IMG_DIR}")
    
    # 1) Read classification data; collect images per disease class
    print("Reading classification annotations...")
    images_per_class = defaultdict(list)
    class_counts = Counter()
    
    try:
        with open(CSV_PATH, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = row["Image Index"].strip()
                findings = row["Finding Labels"].strip()
                
                # Parse disease labels
                disease_labels = parse_disease_labels(findings)
                
                # Add image to each disease class
                for label in disease_labels:
                    images_per_class[label].append(image_name)
                    class_counts[label] += 1
                    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False
    
    # Filter classes with minimum samples
    print(f"\nFound {len(class_counts)} disease classes:")
    filtered_classes = {}
    for class_name, count in class_counts.most_common():
        if count >= MIN_SAMPLES_PER_CLASS:
            filtered_classes[class_name] = count
            print(f"  {class_name}: {count} images")
        else:
            print(f"  {class_name}: {count} images (too few samples)")
    
    print(f"\nUsing {len(filtered_classes)} classes with ≥{MIN_SAMPLES_PER_CLASS} samples")
    
    # 2) Create class directories and copy images
    print("\nCreating class directories and copying images...")
    
    total_copied = 0
    total_missing = 0
    
    for class_name, count in filtered_classes.items():
        # Create class directory
        class_dir = Path(OUT_DIR) / class_name.replace(" ", "_").replace("/", "_")
        class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing class: {class_name}")
        print(f"  Directory: {class_dir}")
        print(f"  Expected images: {count}")
        
        copied_count = 0
        missing_count = 0
        
        # Move images for this class
        for image_name in tqdm(images_per_class[class_name], desc=f"Moving {class_name}"):
            src = Path(ALL_IMG_DIR) / image_name
            dst = class_dir / image_name
            
            if not src.exists():
                missing_count += 1
                continue
                
            if COPY_MODE == "copy":
                shutil.copy2(src, dst)
            elif COPY_MODE == "move":
                shutil.move(str(src), str(dst))
            else:  # symlink
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            
            copied_count += 1
        
        print(f"  Moved: {copied_count}")
        print(f"  Missing: {missing_count}")
        
        total_copied += copied_count
        total_missing += missing_count
    
    # 3) Create classification summary CSV
    print("\nWriting classification summary CSV...")
    summary_csv = Path(OUT_DIR) / "classification_summary.csv"
    
    try:
        with open(summary_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Class Name", "Total Images", "Copied Images", "Missing Images"])
            
            for class_name, count in filtered_classes.items():
                class_dir = Path(OUT_DIR) / class_name.replace(" ", "_").replace("/", "_")
                copied_count = len(list(class_dir.glob("*.png")))
                missing_count = count - copied_count
                writer.writerow([class_name, count, copied_count, missing_count])
                
    except Exception as e:
        print(f"Error writing summary CSV file: {e}")
        return False
    
    # Print final summary
    print(f"\nClassification dataset created successfully!")
    print(f"Output directory: {OUT_DIR}")
    print(f"Classes created: {len(filtered_classes)}")
    print(f"Total images moved: {total_copied}")
    print(f"Total missing images: {total_missing}")
    print(f"Summary CSV: {summary_csv}")
    print(f"\nThis dataset can now be used for classification model training.")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
