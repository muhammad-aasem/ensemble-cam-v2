import csv, os, shutil
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# --- paths (edit as needed)
# Use absolute paths from project root
project_root = Path(__file__).parent.parent.parent
CSV_PATH = project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded" / "data" / "BBox_List_2017.csv"
# Try multiple possible image directories (after extraction)
POSSIBLE_IMG_DIRS = [
    project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded" / "data" / "images" / "images",  # Actual location after extraction
    project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded" / "data" / "images",  # Alternative location
    project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded" / "images",  # Original location
    project_root / "dataset" / "NIH-Chest-X-ray" / "just-downloaded",  # Root of extracted files
]
OUT_DIR = project_root / "dataset" / "NIH-Chest-X-ray" / "BBox"                        # will be created
COPY_MODE = "copy"                                               # "copy" or "symlink"


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


def main():
    """Main function to create BBox subset dataset."""
    print("Creating BBox subset dataset for object detection training...")
    
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
    
    # 1) Read bbox list; collect unique filenames & per-image boxes
    print("Reading BBox annotations...")
    files_with_boxes = set()
    boxes_per_image = defaultdict(list)
    
    try:
        with open(CSV_PATH, newline='', encoding="utf-8") as f:
            # Expected columns: Image Index, Finding Label, x, y, w, h
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["Image Index"].strip()
                label = row["Finding Label"].strip()
                # Parse bbox coordinates from separate columns
                # Convert to int after parsing as float (coordinates are floats in CSV)
                x = int(float(row["Bbox [x"]))
                y = int(float(row["y"]))
                w = int(float(row["w"]))
                h = int(float(row["h]"]))
                files_with_boxes.add(fname)
                boxes_per_image[fname].append({"label": label, "bbox": [x, y, w, h]})
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

    # Validate that we can find some expected images
    print("Validating image availability...")
    sample_files = list(files_with_boxes)[:5]  # Check first 5 files
    found_samples = 0
    for sample_file in sample_files:
        sample_path = Path(ALL_IMG_DIR) / sample_file
        if sample_path.exists():
            found_samples += 1
    
    if found_samples == 0:
        print(f"Warning: None of the expected images found in {ALL_IMG_DIR}")
        print("This might indicate the images haven't been extracted yet.")
        print("Please run dataset_2_extract.py first.")
        return False
    elif found_samples < len(sample_files):
        print(f"Warning: Only {found_samples}/{len(sample_files)} sample images found")
        print("Some images might be missing, but continuing with available ones...")
    else:
        print(f"Validation passed: Found {found_samples}/{len(sample_files)} sample images")

    # 2) Copy only those PNGs with progress tracking
    print(f"Found {len(files_with_boxes)} images with bounding boxes")
    print("Copying images...")
    
    missing, moved = 0, 0
    for fname in tqdm(sorted(files_with_boxes), desc="Copying images"):
        src = Path(ALL_IMG_DIR) / fname
        dst = Path(OUT_DIR) / fname
        if not src.exists():
            missing += 1
            continue
        if COPY_MODE == "copy":
            shutil.copy2(src, dst)
        else:  # symlink
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        moved += 1

    # 3) Save a filtered bbox CSV inside the subset folder
    print("Writing filtered CSV...")
    filtered_csv = Path(OUT_DIR) / "BBox_List_2017.filtered.csv"
    try:
        with open(filtered_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Index", "Finding Label", "x", "y", "w", "h"])
            for fname, items in boxes_per_image.items():
                for it in items:
                    x, y, w, h = it["bbox"]
                    writer.writerow([fname, it["label"], x, y, w, h])
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False
    
    # Print summary
    print(f"\nBBox subset dataset created successfully!")
    print(f"Output directory: {OUT_DIR}")
    print(f"Images with boxes found: {len(files_with_boxes)}")
    print(f"Images copied: {moved}")
    print(f"Missing images: {missing}")
    print(f"Filtered CSV: {filtered_csv}")
    print(f"\nThis subset can now be used for object detection model training.")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
