#!/usr/bin/env python3
"""
Dataset extraction script for NIH Chest X-ray dataset.

This script extracts all zip files from the downloaded dataset and deletes
the zip files after successful extraction. It processes files in the
dataset/NIH-Chest-X-ray/just-downloaded directory.
"""

import os
import zipfile
import sys
from pathlib import Path
from tqdm import tqdm


def extract_zip_files():
    """
    Extract all zip files in the downloaded dataset directory.
    
    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    # Configuration
    # Use absolute path from project root
    project_root = Path(__file__).parent.parent.parent
    dataset_dir = project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded"
    
    # Check if dataset directory exists
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        print("Please run dataset_1_download.py first to download the dataset.")
        return False
    
    # Find all zip files
    zip_files = list(dataset_dir.rglob("*.zip"))
    
    if not zip_files:
        print("No zip files found in the dataset directory.")
        print("Dataset may already be extracted or no zip files present.")
        return True
    
    print(f"Found {len(zip_files)} zip files to extract:")
    for zip_file in zip_files:
        print(f"  - {zip_file.name}")
    
    # Extract each zip file
    extracted_count = 0
    failed_extractions = []
    
    for zip_file in tqdm(zip_files, desc="Extracting zip files"):
        try:
            print(f"\nExtracting: {zip_file.name}")
            
            # Create extraction directory (same as zip file directory)
            extract_dir = zip_file.parent
            
            # Extract the zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Get list of files in zip
                file_list = zip_ref.namelist()
                
                # Extract with progress
                for file_info in tqdm(file_list, desc=f"Extracting {zip_file.name}", leave=False):
                    zip_ref.extract(file_info, extract_dir)
            
            # Delete the zip file after successful extraction
            zip_file.unlink()
            extracted_count += 1
            print(f"Successfully extracted and deleted: {zip_file.name}")
            
        except zipfile.BadZipFile:
            error_msg = f"Bad zip file: {zip_file.name}"
            print(f"{error_msg}")
            failed_extractions.append(error_msg)
        except Exception as e:
            error_msg = f"Error extracting {zip_file.name}: {str(e)}"
            print(f"{error_msg}")
            failed_extractions.append(error_msg)
    
    # Print summary
    print(f"\nExtraction Summary:")
    print(f"Successfully extracted: {extracted_count} files")
    print(f"Failed extractions: {len(failed_extractions)}")
    
    if failed_extractions:
        print("\nFailed extractions:")
        for error in failed_extractions:
            print(f"  - {error}")
        return False
    
    print(f"\nAll zip files extracted successfully!")
    print(f"Extracted files are in: {dataset_dir}")
    
    return True


def main():
    """Main function to run the dataset extraction."""
    print("Starting NIH Chest X-ray dataset extraction...")
    
    success = extract_zip_files()
    
    if success:
        print("Dataset extraction completed successfully!")
        sys.exit(0)
    else:
        print("Dataset extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
