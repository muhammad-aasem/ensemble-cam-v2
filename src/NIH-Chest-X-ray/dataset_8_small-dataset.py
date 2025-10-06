#!/usr/bin/env python3
"""
Small dataset creation script for NIH Chest X-ray dataset.

This script creates a smaller subset of the Classification dataset for development
and testing purposes. It allows you to specify the exact number of images per class
through a data dictionary, enabling quick iteration and testing before running
on the full dataset.

This is particularly useful for:
- Development and debugging
- Quick model testing
- Resource-constrained environments
- Rapid prototyping
"""

import os
import random
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- Configuration (edit as needed)
# Use absolute paths from project root
project_root = Path(__file__).parent.parent.parent
CLASSIFICATION_DIR = project_root / "dataset" / "NIH-Chest-X-ray" / "Classification"
SMALL_DATASET_DIR = project_root / "dataset" / "NIH-Chest-X-ray" / "Classification_SMALL"

# Data dictionary: Class name -> Number of images to include
# Adjust these numbers based on your development needs
CLASS_SAMPLE_COUNTS = {
    'Atelectasis': 150,        # Reduced from ~1000+ to 50
    'Cardiomegaly': 130,       # Reduced from ~500+ to 30
    'Consolidation': 140,      # Reduced from ~800+ to 40
    'Edema': 125,              # Reduced from ~400+ to 25
    'Effusion': 160,           # Reduced from ~1200+ to 60
    'Emphysema': 120,          # Reduced from ~300+ to 20
    'Fibrosis': 115,           # Reduced from ~200+ to 15
    'Hernia': 25,              # Reduced from ~50+ to 5
    'Infiltration': 180,       # Reduced from ~2000+ to 80
    'Mass': 135,               # Reduced from ~600+ to 35
    'No_Finding': 200,        # Reduced from ~3000+ to 100
    'Nodule': 145,             # Reduced from ~800+ to 45
    'Pleural_Thickening': 120, # Reduced from ~300+ to 20
    'Pneumonia': 130,          # Reduced from ~500+ to 30
    'Pneumothorax': 125,       # Reduced from ~400+ to 25
}

# Random seed for reproducible sampling
RANDOM_SEED = 42

# Copy mode: "copy" (duplicate files) or "symlink" (create symbolic links)
COPY_MODE = "copy"  # Use "symlink" to save disk space

# Dataset split configuration
TRAIN_RATIO = 0.7  # 70% for training
VAL_RATIO = 0.2    # 20% for validation
TEST_RATIO = 0.1   # 10% for testing
SPLIT_RANDOM_SEED = 42  # Seed for reproducible splits

# Visualization configuration
FIGURE_SIZE = (15, 10)
BAR_WIDTH = 0.8
COLORS = {
    'train': '#2E8B57',      # Sea Green
    'val': '#4169E1',        # Royal Blue  
    'test': '#DC143C'        # Crimson
}
DPI = 300  # High resolution for publications
FONT_SIZE = 12
TITLE_FONT_SIZE = 16


def get_class_statistics():
    """
    Get statistics about available images per class in the original dataset.
    
    Returns:
        dict: Class name -> count of available images
    """
    stats = {}
    
    if not CLASSIFICATION_DIR.exists():
        print(f"Error: Classification directory not found at {CLASSIFICATION_DIR}")
        return stats
    
    print("Analyzing original dataset...")
    
    for class_name in CLASS_SAMPLE_COUNTS.keys():
        class_dir = CLASSIFICATION_DIR / class_name
        if class_dir.exists():
            # Count both PNG and JPEG files
            png_files = list(class_dir.glob("*.png"))
            jpg_files = list(class_dir.glob("*.jpg"))
            total_files = len(png_files) + len(jpg_files)
            stats[class_name] = total_files
        else:
            stats[class_name] = 0
            print(f"  Warning: Class directory '{class_name}' not found")
    
    return stats


def create_small_dataset():
    """
    Create a small dataset by sampling images from each class.
    
    Returns:
        dict: Statistics about the created small dataset
    """
    print(f"\nCreating small dataset: {SMALL_DATASET_DIR}")
    
    # Get original dataset statistics
    original_stats = get_class_statistics()
    
    # Create small dataset directory
    SMALL_DATASET_DIR.mkdir(exist_ok=True)
    
    # Set random seed for reproducible sampling
    random.seed(RANDOM_SEED)
    
    small_dataset_stats = {}
    total_copied = 0
    total_requested = 0
    
    print(f"\nProcessing classes...")
    print(f"Copy mode: {COPY_MODE}")
    print(f"Random seed: {RANDOM_SEED}")
    
    for class_name, requested_count in CLASS_SAMPLE_COUNTS.items():
        print(f"\n  Processing class: {class_name}")
        
        class_dir = CLASSIFICATION_DIR / class_name
        small_class_dir = SMALL_DATASET_DIR / class_name
        
        if not class_dir.exists():
            print(f"    Class directory not found: {class_dir}")
            small_dataset_stats[class_name] = {
                'requested': requested_count,
                'available': 0,
                'copied': 0,
                'status': 'not_found'
            }
            continue
        
        # Get all image files (PNG and JPEG)
        png_files = list(class_dir.glob("*.png"))
        jpg_files = list(class_dir.glob("*.jpg"))
        all_files = png_files + jpg_files
        
        available_count = len(all_files)
        total_requested += requested_count
        
        print(f"    Available images: {available_count}")
        print(f"    Requested images: {requested_count}")
        
        if available_count == 0:
            print(f"    No images available for class '{class_name}'")
            small_dataset_stats[class_name] = {
                'requested': requested_count,
                'available': 0,
                'copied': 0,
                'status': 'no_images'
            }
            continue
        
        # Determine how many images to actually copy
        actual_count = min(requested_count, available_count)
        
        if actual_count < requested_count:
            print(f"    Only {actual_count} images available (requested {requested_count})")
        
        # Randomly sample images
        sampled_files = random.sample(all_files, actual_count)
        
        # Create class directory in small dataset
        small_class_dir.mkdir(exist_ok=True)
        
        # Copy or symlink images
        copied_count = 0
        for img_file in tqdm(sampled_files, desc=f"    Copying {class_name}", leave=False):
            dst_file = small_class_dir / img_file.name
            
            try:
                if COPY_MODE == "copy":
                    shutil.copy2(img_file, dst_file)
                else:  # symlink
                    if dst_file.exists():
                        dst_file.unlink()
                    dst_file.symlink_to(img_file.resolve())
                
                copied_count += 1
                total_copied += 1
                
            except Exception as e:
                print(f"    Error copying {img_file.name}: {e}")
        
        print(f"    Copied {copied_count} images")
        
        small_dataset_stats[class_name] = {
            'requested': requested_count,
            'available': available_count,
            'copied': copied_count,
            'status': 'success'
        }
    
    return small_dataset_stats, total_copied, total_requested


def create_dataset_split_csv():
    """
    Create dataset_split.csv for the small dataset with train/val/test splits.
    
    Returns:
        dict: Statistics about the split creation
    """
    print(f"\nCreating dataset split for small dataset...")
    
    # Collect all image paths and labels from the small dataset
    image_paths = []
    labels = []
    class_names = []
    
    # Class name to index mapping
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_SAMPLE_COUNTS.keys())}
    
    for class_name in CLASS_SAMPLE_COUNTS.keys():
        class_dir = SMALL_DATASET_DIR / class_name
        if not class_dir.exists():
            continue
        
        # Get all image files (PNG and JPEG)
        png_files = list(class_dir.glob("*.png"))
        jpg_files = list(class_dir.glob("*.jpg"))
        all_files = png_files + jpg_files
        
        for img_file in all_files:
            image_paths.append(str(img_file))
            labels.append(class_to_idx[class_name])
            class_names.append(class_name)
    
    if not image_paths:
        print("No images found in small dataset")
        return {"total": 0, "train": 0, "val": 0, "test": 0}
    
    print(f"  Found {len(image_paths)} total images")
    
    # Create DataFrame
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels,
        'class_name': class_names
    })
    
    # Check if we have enough samples for stratified splitting
    min_samples_per_class = df['label'].value_counts().min()
    
    if min_samples_per_class < 2:
        print(f"  Warning: Some classes have < 2 samples, using random split instead of stratified")
        # Use random split for small datasets
        train_df, temp_df = train_test_split(
            df, test_size=(VAL_RATIO + TEST_RATIO), random_state=SPLIT_RANDOM_SEED
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), 
            random_state=SPLIT_RANDOM_SEED
        )
    else:
        # Stratified split: 70% train, 20% val, 10% test
        train_df, temp_df = train_test_split(
            df, test_size=(VAL_RATIO + TEST_RATIO), stratify=df['label'], 
            random_state=SPLIT_RANDOM_SEED
        )
        
        # Check if temp_df has enough samples for stratified splitting
        temp_min_samples = temp_df['label'].value_counts().min()
        
        if temp_min_samples < 2:
            print(f"  Warning: Some classes have < 2 samples in val/test set, using random split")
            val_df, test_df = train_test_split(
                temp_df, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), 
                random_state=SPLIT_RANDOM_SEED
            )
        else:
            val_df, test_df = train_test_split(
                temp_df, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), 
                stratify=temp_df['label'], random_state=SPLIT_RANDOM_SEED
            )
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Combine and save
    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_file = SMALL_DATASET_DIR / 'dataset_split.csv'
    split_df.to_csv(split_file, index=False)
    
    print(f"  Dataset split created:")
    print(f"    Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"    Split file: {split_file}")
    
    return {
        "total": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df)
    }


def create_small_dataset_summary(stats, total_copied, total_requested, split_stats):
    """
    Create a summary CSV file for the small dataset.
    
    Args:
        stats: Statistics dictionary from create_small_dataset
        total_copied: Total number of images copied
        total_requested: Total number of images requested
        split_stats: Statistics from dataset split creation
    """
    summary_file = SMALL_DATASET_DIR / "small_dataset_summary.csv"
    
    print(f"\nCreating summary file: {summary_file}")
    
    with open(summary_file, 'w') as f:
        f.write("Class Name,Requested Images,Available Images,Copied Images,Status\n")
        
        for class_name, class_stats in stats.items():
            f.write(f"{class_name},{class_stats['requested']},"
                   f"{class_stats['available']},{class_stats['copied']},"
                   f"{class_stats['status']}\n")
        
        # Add totals row
        f.write(f"TOTAL,{total_requested},,{total_copied},\n")
        
        # Add split information
        f.write(f"\nDataset Split Information\n")
        f.write(f"Train Samples,{split_stats['train']}\n")
        f.write(f"Validation Samples,{split_stats['val']}\n")
        f.write(f"Test Samples,{split_stats['test']}\n")
        f.write(f"Total Samples,{split_stats['total']}\n")
        f.write(f"Split Ratios,{TRAIN_RATIO:.1%}/{VAL_RATIO:.1%}/{TEST_RATIO:.1%}\n")
        f.write(f"Split Seed,{SPLIT_RANDOM_SEED}\n")
    
    print(f"Summary file created")


def print_statistics(original_stats, small_dataset_stats, total_copied, total_requested, split_stats):
    """
    Print detailed statistics about the dataset creation process.
    
    Args:
        original_stats: Statistics from original dataset
        small_dataset_stats: Statistics from small dataset creation
        total_copied: Total number of images copied
        total_requested: Total number of images requested
        split_stats: Statistics from dataset split creation
    """
    print(f"\nDataset Creation Statistics")
    print("=" * 60)
    
    # Calculate totals
    total_available = sum(original_stats.values())
    
    print(f"Small dataset directory: {SMALL_DATASET_DIR}")
    print(f"Original dataset directory: {CLASSIFICATION_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Copy mode: {COPY_MODE}")
    print(f"Split ratios: {TRAIN_RATIO:.1%}/{VAL_RATIO:.1%}/{TEST_RATIO:.1%}")
    print(f"Split seed: {SPLIT_RANDOM_SEED}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total images requested: {total_requested:,}")
    print(f"  Total images available: {total_available:,}")
    print(f"  Total images copied: {total_copied:,}")
    print(f"  Success rate: {(total_copied/total_requested)*100:.1f}%")
    print(f"  Dataset reduction: {(1-total_copied/total_available)*100:.1f}%")
    
    print(f"\nDataset Split Statistics:")
    print(f"  Train samples: {split_stats['train']:,} ({split_stats['train']/split_stats['total']*100:.1f}%)")
    print(f"  Validation samples: {split_stats['val']:,} ({split_stats['val']/split_stats['total']*100:.1f}%)")
    print(f"  Test samples: {split_stats['test']:,} ({split_stats['test']/split_stats['total']*100:.1f}%)")
    print(f"  Total samples: {split_stats['total']:,}")
    
    print(f"\nPer-Class Statistics:")
    print(f"{'Class Name':<20} {'Requested':<10} {'Available':<10} {'Copied':<8} {'Status':<12}")
    print("-" * 70)
    
    for class_name in CLASS_SAMPLE_COUNTS.keys():
        if class_name in small_dataset_stats:
            stats = small_dataset_stats[class_name]
            print(f"{class_name:<20} {stats['requested']:<10} "
                 f"{stats['available']:<10} {stats['copied']:<8} {stats['status']:<12}")
    
    print(f"\nStorage Information:")
    if COPY_MODE == "copy":
        print(f"  Mode: File copying (duplicates files)")
        print(f"  Storage impact: Additional {total_copied} files")
    else:
        print(f"  Mode: Symbolic links (saves space)")
        print(f"  Storage impact: Minimal (links only)")
    
    print(f"\nGenerated Files:")
    print(f"  Dataset split: {SMALL_DATASET_DIR / 'dataset_split.csv'}")
    print(f"  Summary file: {SMALL_DATASET_DIR / 'small_dataset_summary.csv'}")


def create_bar_chart(analysis, dataset_name, output_path):
    """
    Create a bar chart showing class distribution across splits.
    
    Args:
        analysis: Analysis results from analyze_dataset_distribution
        dataset_name: Name of the dataset for the title
        output_path: Path to save the chart
    """
    if analysis is None:
        return False
    
    distribution = analysis['distribution']
    splits = analysis['splits']
    
    # Set up the plot
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Create bar chart
    x = range(len(distribution))
    width = BAR_WIDTH / len(splits)
    
    bars = []
    for i, split in enumerate(splits):
        if split in distribution.columns:
            bar = ax.bar([pos + i * width for pos in x], 
                        distribution[split], 
                        width, 
                        label=split.capitalize(),
                        color=COLORS[split],
                        alpha=0.8)
            bars.append(bar)
    
    # Customize the plot
    ax.set_xlabel('Classes', fontsize=FONT_SIZE)
    ax.set_ylabel('Number of Samples', fontsize=FONT_SIZE)
    ax.set_title(f'Dataset Distribution: {dataset_name}\n'
                f'Total Samples: {analysis["total_samples"]:,} | '
                f'Classes: {analysis["num_classes"]}', 
                fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks([pos + width for pos in x])
    ax.set_xticklabels(distribution.index, rotation=45, ha='right')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, split in enumerate(splits):
        if split in distribution.columns:
            for j, (idx, row) in enumerate(distribution.iterrows()):
                value = row[split]
                if value > 0:  # Only show labels for non-zero values
                    ax.text(j + i * width, value + 0.5, str(int(value)), 
                           ha='center', va='bottom', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Bar chart saved: {output_path}")
    return True


def create_summary_table(analysis, dataset_name, output_path):
    """
    Create a summary table with detailed statistics.
    
    Args:
        analysis: Analysis results from analyze_dataset_distribution
        dataset_name: Name of the dataset for the title
        output_path: Path to save the table
    """
    if analysis is None:
        return False
    
    distribution = analysis['distribution']
    
    # Create summary table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for class_name, row in distribution.iterrows():
        table_data.append([
            class_name,
            int(row['train']),
            int(row['val']),
            int(row['test']),
            int(row['total']),
            f"{row['train_pct']:.1f}%",
            f"{row['val_pct']:.1f}%",
            f"{row['test_pct']:.1f}%"
        ])
    
    # Add totals row
    totals = distribution.sum()
    table_data.append([
        'TOTAL',
        int(totals['train']),
        int(totals['val']),
        int(totals['test']),
        int(totals['total']),
        f"{(totals['train']/totals['total']*100):.1f}%",
        f"{(totals['val']/totals['total']*100):.1f}%",
        f"{(totals['test']/totals['total']*100):.1f}%"
    ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Class', 'Train', 'Val', 'Test', 'Total', 
                              'Train %', 'Val %', 'Test %'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color the header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color the totals row
    last_row = len(table_data)
    for i in range(len(table_data[0])):
        table[(last_row, i)].set_facecolor('#FFC107')
        table[(last_row, i)].set_text_props(weight='bold')
    
    # Set title
    ax.set_title(f'Dataset Summary: {dataset_name}\n'
                f'Total Samples: {analysis["total_samples"]:,} | '
                f'Classes: {analysis["num_classes"]}', 
                fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=20)
    
    # Save the table
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Summary table saved: {output_path}")
    return True


def analyze_dataset_distribution(df):
    """
    Analyze the distribution of classes across train/val/test splits.
    
    Args:
        df: DataFrame with dataset split information
        
    Returns:
        dict: Analysis results
    """
    if df is None:
        return None
    
    # Count samples per class and split
    distribution = df.groupby(['class_name', 'split']).size().unstack(fill_value=0)
    
    # Ensure all split columns exist
    for split in ['train', 'val', 'test']:
        if split not in distribution.columns:
            distribution[split] = 0
    
    # Calculate totals
    distribution['total'] = distribution.sum(axis=1)
    distribution = distribution.sort_values('total', ascending=True)
    
    # Calculate percentages
    total_samples = distribution['total'].sum()
    distribution['train_pct'] = (distribution['train'] / distribution['total'] * 100).round(1)
    distribution['val_pct'] = (distribution['val'] / distribution['total'] * 100).round(1)
    distribution['test_pct'] = (distribution['test'] / distribution['total'] * 100).round(1)
    
    return {
        'distribution': distribution,
        'total_samples': total_samples,
        'num_classes': len(distribution),
        'splits': ['train', 'val', 'test']
    }


def generate_dataset_visualizations():
    """
    Generate bar chart and summary table visualizations for the small dataset.
    
    Returns:
        bool: True if visualizations were generated successfully
    """
    print(f"\nGenerating dataset visualizations...")
    
    # Load the dataset split CSV
    split_file = SMALL_DATASET_DIR / 'dataset_split.csv'
    if not split_file.exists():
        print(f"Error: dataset_split.csv not found in {SMALL_DATASET_DIR}")
        return False
    
    try:
        df = pd.read_csv(split_file)
        print(f"Loaded dataset split: {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset split: {e}")
        return False
    
    # Analyze distribution
    analysis = analyze_dataset_distribution(df)
    if analysis is None:
        return False
    
    # Generate visualizations
    dataset_name = "Classification_SMALL"
    
    # Create bar chart
    bar_chart_path = SMALL_DATASET_DIR / f"{dataset_name}_distribution_bar_chart.png"
    create_bar_chart(analysis, dataset_name, bar_chart_path)
    
    # Create summary table
    table_path = SMALL_DATASET_DIR / f"{dataset_name}_summary_table.png"
    create_summary_table(analysis, dataset_name, table_path)
    
    print(f"\nGenerated visualization files:")
    print(f"  Bar chart: {bar_chart_path}")
    print(f"  Summary table: {table_path}")
    
    return True


def main():
    """Main function to create the small dataset."""
    print("NIH Chest X-ray Dataset - Small Dataset Creation")
    print("=" * 60)
    print("This script creates a smaller subset of the Classification dataset")
    print("for development and testing purposes.")
    
    # Validate original dataset exists
    if not CLASSIFICATION_DIR.exists():
        print(f"Error: Classification directory not found at {CLASSIFICATION_DIR}")
        print("Please run dataset_4_classification.py first to create the classification dataset.")
        return False
    
    # Get original dataset statistics
    original_stats = get_class_statistics()
    
    if not original_stats:
        print("No classes found in the original dataset")
        return False
    
    # Create small dataset
    small_dataset_stats, total_copied, total_requested = create_small_dataset()
    
    # Create dataset split CSV
    split_stats = create_dataset_split_csv()
    
    # Create summary file
    create_small_dataset_summary(small_dataset_stats, total_copied, total_requested, split_stats)
    
    # Generate visualizations
    generate_dataset_visualizations()
    
    # Print statistics
    print_statistics(original_stats, small_dataset_stats, total_copied, total_requested, split_stats)
    
    print(f"\nSmall dataset creation completed successfully!")
    print(f"Small dataset location: {SMALL_DATASET_DIR}")
    print(f"Summary file: {SMALL_DATASET_DIR / 'small_dataset_summary.csv'}")
    print(f"Dataset split file: {SMALL_DATASET_DIR / 'dataset_split.csv'}")
    print(f"Bar chart: {SMALL_DATASET_DIR / 'Classification_SMALL_distribution_bar_chart.png'}")
    print(f"Summary table: {SMALL_DATASET_DIR / 'Classification_SMALL_summary_table.png'}")
    
    print(f"\nUsage Tips:")
    print(f"  - Use this small dataset for development and testing")
    print(f"  - Adjust CLASS_SAMPLE_COUNTS in the script to change sample sizes")
    print(f"  - Set COPY_MODE to 'symlink' to save disk space")
    print(f"  - Use the same RANDOM_SEED for reproducible results")
    print(f"  - The dataset_split.csv provides train/val/test splits for training")
    print(f"  - Split ratios: {TRAIN_RATIO:.1%} train, {VAL_RATIO:.1%} val, {TEST_RATIO:.1%} test")
    print(f"  - Visualizations show the distribution of classes across splits")
    
    return total_copied > 0


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
