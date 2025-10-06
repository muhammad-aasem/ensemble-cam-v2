#!/usr/bin/env python3
"""
Dataset cleanup script for NIH Chest X-ray dataset.

This script removes the just-downloaded directory after all processing is complete.
This helps free up storage space by removing the original downloaded files
once they have been processed into the BBox and Classification datasets.
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# --- paths (edit as needed)
# Use absolute paths from project root
project_root = Path(__file__).parent.parent.parent
JUST_DOWNLOADED_DIR = project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded"


def get_directory_size(path):
    """
    Calculate the total size of a directory in bytes.
    
    Args:
        path: Path to the directory
        
    Returns:
        int: Total size in bytes
    """
    total_size = 0
    try:
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except (OSError, IOError):
        pass
    return total_size


def format_size(size_bytes):
    """
    Convert bytes to human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Human readable size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def main():
    """Main function to clean up the just-downloaded directory."""
    print("NIH Chest X-ray Dataset Cleanup")
    print("=" * 50)
    
    # Check if just-downloaded directory exists
    if not JUST_DOWNLOADED_DIR.exists():
        print(f"Directory not found: {JUST_DOWNLOADED_DIR}")
        print("Nothing to clean up!")
        return True
    
    # Calculate directory size before deletion
    print("Calculating directory size...")
    total_size = get_directory_size(JUST_DOWNLOADED_DIR)
    size_str = format_size(total_size)
    
    print(f"Directory to delete: {JUST_DOWNLOADED_DIR}")
    print(f"Total size: {size_str}")
    
    # Count files
    file_count = 0
    try:
        file_count = len(list(JUST_DOWNLOADED_DIR.rglob('*')))
    except (OSError, IOError):
        pass
    
    print(f"Total files: {file_count:,}")
    
    # Safety check - verify that BBox and Classification datasets exist
    bbox_dir = project_root / "dataset" / "NIH-Chest-X-ray" / "BBox"
    classification_dir = project_root / "dataset" / "NIH-Chest-X-ray" / "Classification"
    
    if not bbox_dir.exists():
        print(f"BBox dataset not found: {bbox_dir}")
        print("Please run dataset_3_subset-bbox.py first.")
        return False
    
    if not classification_dir.exists():
        print(f"Classification dataset not found: {classification_dir}")
        print("Please run dataset_4_classification.py first.")
        return False
    
    print("Safety checks passed - both BBox and Classification datasets exist")
    
    # Delete the directory
    print(f"\nDeleting directory: {JUST_DOWNLOADED_DIR}")
    
    try:
        # Use tqdm for progress indication
        with tqdm(total=file_count, desc="Deleting files", unit="files") as pbar:
            def update_progress(*args):
                pbar.update(1)
            
            # Monkey patch shutil.rmtree to show progress
            original_rmtree = shutil.rmtree
            def rmtree_with_progress(path, ignore_errors=False, onerror=None):
                if onerror is None:
                    def onerror(func, path, exc_info):
                        if not ignore_errors:
                            raise exc_info[1]
                else:
                    onerror = onerror
                
                # Count files first
                file_count = 0
                for root, dirs, files in os.walk(path):
                    file_count += len(files)
                
                # Delete with progress
                for root, dirs, files in os.walk(path, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                            update_progress()
                        except OSError:
                            pass
                    for dir in dirs:
                        try:
                            os.rmdir(os.path.join(root, dir))
                        except OSError:
                            pass
                try:
                    os.rmdir(path)
                except OSError:
                    pass
            
            # Temporarily replace shutil.rmtree
            import os
            shutil.rmtree = rmtree_with_progress
            
            # Delete the directory
            shutil.rmtree(JUST_DOWNLOADED_DIR, ignore_errors=True)
            
            # Restore original function
            shutil.rmtree = original_rmtree
            
    except Exception as e:
        print(f"Error during deletion: {e}")
        return False
    
    # Verify deletion
    if not JUST_DOWNLOADED_DIR.exists():
        print(f"\nSuccessfully deleted: {JUST_DOWNLOADED_DIR}")
        print(f"Freed up: {size_str}")
        print(f"Files removed: {file_count:,}")
        print("\nCleanup completed successfully!")
        print("\nYour dataset structure is now:")
        print("  dataset/NIH-Chest-X-ray/")
        print("  ├── BBox/                    (Object detection subset)")
        print("  └── Classification/           (Classification dataset)")
        return True
    else:
        print(f"Directory still exists: {JUST_DOWNLOADED_DIR}")
        print("Deletion may have failed.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
