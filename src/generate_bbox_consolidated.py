#!/usr/bin/env python3
"""
Generate consolidated bounding box from multiple binary masks (P3).
Merges multiple masks and creates JSON with bounding box coordinates and visualization.

Usage:
    python src/generate_bbox_consolidated.py \
        --input-sources mask1_P3.png mask2_P3.png mask3_P3.png \
        --method merge-masking \
        --target-class "Cardiomegaly" \
        --ground-truth "Cardiomegaly,277,459,540,301"
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def merge_masks(mask_paths, method='merge-masking'):
    """
    Merge multiple binary masks into one consolidated mask.
    
    Args:
        mask_paths: List of paths to binary mask images
        method: Aggregation method ('merge-masking' for now)
    
    Returns:
        Consolidated mask as numpy array
    """
    if method != 'merge-masking':
        raise ValueError(f"Unsupported method: {method}. Only 'merge-masking' is currently supported.")
    
    masks = []
    reference_shape = None
    
    # Load all masks
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask image: {mask_path}")
        
        if reference_shape is None:
            reference_shape = mask.shape
        elif mask.shape != reference_shape:
            raise ValueError(f"Mask shape mismatch: {mask_path} has shape {mask.shape}, expected {reference_shape}")
        
        masks.append(mask)
    
    if not masks:
        raise ValueError("No masks provided")
    
    # Merge-masking: Combine all non-zero pixels (OR operation)
    consolidated = np.zeros_like(masks[0])
    for mask in masks:
        consolidated = cv2.bitwise_or(consolidated, mask)
    
    return consolidated


def find_bounding_box(mask):
    """
    Find tight bounding box around non-zero regions in a binary mask.
    
    Args:
        mask: Binary mask image (numpy array)
    
    Returns:
        Dictionary with bounding box coordinates or None if no regions found.
    """
    # Find all non-zero pixels
    coords = cv2.findNonZero(mask)
    
    if coords is None:
        print("Warning: No non-zero regions found in consolidated mask")
        return None
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)
    
    return {
        'x': int(x),
        'y': int(y),
        'width': int(w),
        'height': int(h),
        'x2': int(x + w),
        'y2': int(y + h)
    }


def draw_bounding_box(background_image, bbox, color=(0, 255, 0), thickness=3, label=None):
    """
    Draw bounding box on a background image.
    
    Args:
        background_image: Background image (numpy array) to draw on
        bbox: Bounding box dictionary from find_bounding_box()
        color: BGR color tuple (default: green)
        thickness: Line thickness in pixels (default: 3)
        label: Optional text label to draw above the box
    
    Returns:
        Image with bounding box drawn
    """
    # Make a copy to avoid modifying the original
    img = background_image.copy()
    
    # Draw rectangle
    cv2.rectangle(
        img,
        (bbox['x'], bbox['y']),
        (bbox['x2'], bbox['y2']),
        color,
        thickness
    )
    
    # Draw label if provided
    if label:
        # Position label above the box
        label_pos = (bbox['x'], bbox['y'] - 10 if bbox['y'] > 20 else bbox['y'] + 20)
        cv2.putText(
            img,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    
    return img


def draw_bounding_boxes(background_image, predicted_bbox, ground_truth_bbox=None,
                       pred_color=(0, 255, 0), gt_color=(0, 0, 255), thickness=3):
    """
    Draw both predicted and ground truth bounding boxes.
    
    Args:
        background_image: Background image to draw on
        predicted_bbox: Predicted bounding box dictionary
        ground_truth_bbox: Ground truth bounding box dictionary (optional)
        pred_color: Color for predicted box (default: green)
        gt_color: Color for ground truth box (default: red)
        thickness: Line thickness
    
    Returns:
        Image with both boxes drawn
    """
    img = background_image.copy()
    
    # Draw predicted box
    img = draw_bounding_box(img, predicted_bbox, pred_color, thickness, "Predicted")
    
    # Draw ground truth box if provided
    if ground_truth_bbox:
        img = draw_bounding_box(img, ground_truth_bbox, gt_color, thickness, "Ground Truth")
    
    return img


def parse_ground_truth(gt_string):
    """
    Parse ground truth CSV string.
    
    Args:
        gt_string: CSV string "Label,x,y,w,h"
    
    Returns:
        Dictionary with ground truth data
    """
    parts = gt_string.split(',')
    if len(parts) != 5:
        raise ValueError(f"Ground truth must have 5 comma-separated values (Label,x,y,w,h), got {len(parts)}")
    
    label = parts[0].strip()
    x, y, w, h = map(int, parts[1:])
    
    return {
        'label': label,
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'x2': x + w,
        'y2': y + h
    }


def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1_min, y1_min = bbox1['x'], bbox1['y']
    x1_max, y1_max = bbox1['x2'], bbox1['y2']
    x2_min, y2_min = bbox2['x'], bbox2['y']
    x2_max, y2_max = bbox2['x2'], bbox2['y2']
    
    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    intersection = inter_width * inter_height
    
    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    # IoU
    iou = intersection / union if union > 0 else 0.0
    return float(iou)


def compute_dice_coefficient(bbox1, bbox2):
    """Compute Dice coefficient between two bounding boxes."""
    x1_min, y1_min = bbox1['x'], bbox1['y']
    x1_max, y1_max = bbox1['x2'], bbox1['y2']
    x2_min, y2_min = bbox2['x'], bbox2['y']
    x2_max, y2_max = bbox2['x2'], bbox2['y2']
    
    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    intersection = inter_width * inter_height
    
    # Areas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Dice = 2 * |A ∩ B| / (|A| + |B|)
    dice = (2.0 * intersection) / (area1 + area2) if (area1 + area2) > 0 else 0.0
    return float(dice)


def compute_pixel_accuracy(bbox1, bbox2, image_shape):
    """Compute pixel-level accuracy between two bounding boxes."""
    h, w = image_shape[:2]
    total_pixels = h * w
    
    # True Positives: Intersection
    x1_min, y1_min = bbox1['x'], bbox1['y']
    x1_max, y1_max = bbox1['x2'], bbox1['y2']
    x2_min, y2_min = bbox2['x'], bbox2['y']
    x2_max, y2_max = bbox2['x2'], bbox2['y2']
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    tp = inter_width * inter_height
    
    # Areas
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # True Negatives: Pixels outside both boxes
    tn = total_pixels - bbox1_area - bbox2_area + tp
    
    # Pixel Accuracy = (TP + TN) / Total
    accuracy = (tp + tn) / total_pixels
    return float(accuracy)


def compute_colloc(iou, threshold):
    """Compute Correct Localization (ColLoc) metric."""
    return 1.0 if iou >= threshold else 0.0


def compute_precision_recall(bbox_pred, bbox_gt, iou_threshold=0.5):
    """Compute precision and recall for object detection."""
    iou = compute_iou(bbox_pred, bbox_gt)
    
    if iou >= iou_threshold:
        precision = 1.0
        recall = 1.0
    else:
        precision = 0.0
        recall = 0.0
    
    return precision, recall


def compute_map(bbox_pred, bbox_gt):
    """Compute mean Average Precision (mAP) at different IoU thresholds."""
    iou = compute_iou(bbox_pred, bbox_gt)
    
    # COCO-style mAP: IoU thresholds from 0.5 to 0.95 in steps of 0.05
    thresholds = [0.5 + i * 0.05 for i in range(10)]
    
    ap_scores = []
    for thresh in thresholds:
        ap_scores.append(1.0 if iou >= thresh else 0.0)
    
    map_value = sum(ap_scores) / len(ap_scores)
    
    return {
        'mAP@[.5:.95]': round(map_value, 4),
        'AP@.5': round(1.0 if iou >= 0.5 else 0.0, 4),
        'AP@.75': round(1.0 if iou >= 0.75 else 0.0, 4)
    }


def save_consolidated_json(output_path, bbox, source_masks, method, target_class, 
                           image_shape, ground_truth=None, metrics=None):
    """
    Save consolidated bounding box data to JSON.
    
    Args:
        output_path: Path to save JSON file
        bbox: Bounding box dictionary
        source_masks: List of source mask paths
        method: Consolidation method used
        target_class: Target class name
        image_shape: Shape of the image (h, w)
        ground_truth: Optional ground truth dictionary
        metrics: Optional metrics dictionary
    """
    data = {
        'consolidation': {
            'method': method,
            'target_class': target_class,
            'source_masks': [str(p) for p in source_masks],
            'num_sources': len(source_masks)
        },
        'bounding_box': {
            'x': bbox['x'],
            'y': bbox['y'],
            'width': bbox['width'],
            'height': bbox['height'],
            'x2': bbox['x2'],
            'y2': bbox['y2']
        },
        'image_dimensions': {
            'height': int(image_shape[0]),
            'width': int(image_shape[1])
        },
        'format': 'x, y, width, height'
    }
    
    if ground_truth:
        data['ground_truth'] = ground_truth
    
    if metrics:
        data['metrics'] = metrics
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Generate consolidated bounding box from multiple binary masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage with merge-masking
  python src/generate_bbox_consolidated.py \\
      --input-sources OUTPUT/img/model1_Class_P3.png OUTPUT/img/model2_Class_P3.png \\
      --method merge-masking \\
      --target-class "Cardiomegaly"

  # With ground truth for evaluation
  python src/generate_bbox_consolidated.py \\
      --input-sources OUTPUT/img/model1_Class_P3.png OUTPUT/img/model2_Class_P3.png \\
      --method merge-masking \\
      --target-class "Cardiomegaly" \\
      --ground-truth "Cardiomegaly,277,459,540,301"

Output files:
  - {target_class}_consolidated.png        : Consolidated mask image
  - {target_class}_consolidated_bbox.png   : Bounding box visualization
  - {target_class}_consolidated_bbox.json  : Bounding box data and metrics
        '''
    )
    
    parser.add_argument(
        '--input-sources',
        nargs='+',
        required=True,
        help='Paths to input binary mask images (P3.png files)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='merge-masking',
        choices=['merge-masking'],
        help='Consolidation method (default: merge-masking)'
    )
    
    parser.add_argument(
        '--target-class',
        type=str,
        required=True,
        help='Target class name (e.g., "Cardiomegaly")'
    )
    
    parser.add_argument(
        '--ground-truth',
        type=str,
        help='Ground truth bounding box as CSV: "Label,x,y,w,h"'
    )
    
    parser.add_argument(
        '--color',
        type=int,
        nargs=3,
        default=[0, 255, 0],
        metavar=('B', 'G', 'R'),
        help='Bounding box color in BGR format (default: 0 255 0 for green)'
    )
    
    parser.add_argument(
        '--thickness',
        type=int,
        default=3,
        help='Bounding box line thickness in pixels (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Convert input sources to Path objects
    mask_paths = [Path(p) for p in args.input_sources]
    
    # Validate input files
    for mask_path in mask_paths:
        if not mask_path.exists():
            print(f"❌ Error: Mask file not found: {mask_path}")
            return 1
    
    # Determine output directory (use first mask's directory)
    output_dir = mask_paths[0].parent
    
    # Create safe filename for target class
    safe_class = args.target_class.replace(' ', '_').replace('/', '_')
    
    # Output paths
    consolidated_mask_path = output_dir / f"{safe_class}_consolidated.png"
    bbox_image_path = output_dir / f"{safe_class}_consolidated_bbox.png"
    bbox_json_path = output_dir / f"{safe_class}_consolidated_bbox.json"
    
    print(f"Input masks: {len(mask_paths)} files")
    for i, p in enumerate(mask_paths, 1):
        print(f"  {i}. {p.name}")
    print(f"Consolidation method: {args.method}")
    print(f"Target class: {args.target_class}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Merge masks
        print(f"\n{'='*60}")
        print(f"Step 1: Consolidating masks using '{args.method}'...")
        print(f"{'='*60}")
        consolidated_mask = merge_masks(mask_paths, method=args.method)
        
        # Save consolidated mask
        cv2.imwrite(str(consolidated_mask_path), consolidated_mask)
        print(f"✓ Consolidated mask saved: {consolidated_mask_path.name}")
        
        # Step 2: Find bounding box
        print(f"\n{'='*60}")
        print(f"Step 2: Finding bounding box...")
        print(f"{'='*60}")
        bbox = find_bounding_box(consolidated_mask)
        
        if bbox is None:
            print("❌ Error: No bounding box found in consolidated mask")
            return 1
        
        print(f"Bounding box found:")
        print(f"  Position: ({bbox['x']}, {bbox['y']})")
        print(f"  Size: {bbox['width']} × {bbox['height']} pixels")
        print(f"  Bottom-right: ({bbox['x2']}, {bbox['y2']})")
        
        # Step 3: Parse ground truth and compute metrics if provided
        ground_truth = None
        metrics = None
        
        if args.ground_truth:
            print(f"\n{'='*60}")
            print(f"Step 3: Parsing ground truth and computing metrics...")
            print(f"{'='*60}")
            try:
                ground_truth = parse_ground_truth(args.ground_truth)
                print(f"Ground truth bbox:")
                print(f"  Label: {ground_truth['label']}")
                print(f"  Position: ({ground_truth['x']}, {ground_truth['y']})")
                print(f"  Size: {ground_truth['width']} × {ground_truth['height']} pixels")
                print(f"  Bottom-right: ({ground_truth['x2']}, {ground_truth['y2']})")
                
                # Compute metrics
                iou = compute_iou(bbox, ground_truth)
                dice = compute_dice_coefficient(bbox, ground_truth)
                pixel_acc = compute_pixel_accuracy(bbox, ground_truth, consolidated_mask.shape)
                precision, recall = compute_precision_recall(bbox, ground_truth, iou_threshold=0.5)
                map_scores = compute_map(bbox, ground_truth)
                colloc_01 = compute_colloc(iou, threshold=0.1)
                colloc_03 = compute_colloc(iou, threshold=0.3)
                
                metrics = {
                    'IoU': round(iou, 4),
                    'Dice': round(dice, 4),
                    'pixel_accuracy': round(pixel_acc, 4),
                    'precision@0.5': round(precision, 4),
                    'recall@0.5': round(recall, 4),
                    'ColLoc@0.1': round(colloc_01, 4),
                    'ColLoc@0.3': round(colloc_03, 4),
                    'mAP': map_scores
                }
                
                print(f"\nMetrics:")
                print(f"  IoU: {metrics['IoU']:.4f}")
                print(f"  Dice Coefficient: {metrics['Dice']:.4f}")
                print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
                print(f"  Precision@0.5: {metrics['precision@0.5']:.4f}")
                print(f"  Recall@0.5: {metrics['recall@0.5']:.4f}")
                print(f"  ColLoc@0.1: {metrics['ColLoc@0.1']:.4f}")
                print(f"  ColLoc@0.3: {metrics['ColLoc@0.3']:.4f}")
                print(f"  mAP@[.5:.95]: {metrics['mAP']['mAP@[.5:.95]']:.4f}")
                print(f"  AP@0.5: {metrics['mAP']['AP@.5']:.4f}")
                print(f"  AP@0.75: {metrics['mAP']['AP@.75']:.4f}")
                
            except ValueError as e:
                print(f"⚠️  Warning: Failed to parse ground truth: {e}")
                ground_truth = None
                metrics = None
        
        # Step 4: Save JSON
        step_num = 4 if args.ground_truth else 3
        print(f"\n{'='*60}")
        print(f"Step {step_num}: Saving bounding box data...")
        print(f"{'='*60}")
        save_consolidated_json(
            bbox_json_path,
            bbox,
            mask_paths,
            args.method,
            args.target_class,
            consolidated_mask.shape,
            ground_truth,
            metrics
        )
        print(f"✓ JSON saved: {bbox_json_path.name}")
        
        # Step 5: Generate visualization
        step_num += 1
        print(f"\n{'='*60}")
        print(f"Step {step_num}: Generating visualization...")
        print(f"{'='*60}")
        
        # Load background image (_input_image_.png)
        input_image_path = output_dir / "_input_image_.png"
        if input_image_path.exists():
            print(f"Loading background image: {input_image_path.name}")
            background_image = cv2.imread(str(input_image_path))
            if background_image is None:
                print(f"  ⚠️  Warning: Could not read background image, using white canvas")
                background_image = np.ones((consolidated_mask.shape[0], consolidated_mask.shape[1], 3), dtype=np.uint8) * 255
        else:
            print(f"  ⚠️  Warning: Background image not found at {input_image_path}, using white canvas")
            background_image = np.ones((consolidated_mask.shape[0], consolidated_mask.shape[1], 3), dtype=np.uint8) * 255
        
        # Draw bounding boxes
        if ground_truth is not None:
            # Draw both predicted (green) and ground truth (red) bounding boxes
            viz_image = draw_bounding_boxes(
                background_image,
                predicted_bbox=bbox,
                ground_truth_bbox=ground_truth,
                pred_color=tuple(args.color),
                gt_color=(0, 0, 255),  # Red for ground truth
                thickness=args.thickness
            )
            print(f"  - Predicted bbox: Green")
            print(f"  - Ground truth bbox: Red")
        else:
            # Draw only predicted bounding box
            viz_image = draw_bounding_box(
                background_image,
                bbox,
                color=tuple(args.color),
                thickness=args.thickness,
                label="Consolidated"
            )
        
        cv2.imwrite(str(bbox_image_path), viz_image)
        print(f"✓ Visualization saved: {bbox_image_path.name}")
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"✅ Consolidated bounding box generation complete!")
        print(f"{'='*60}")
        print(f"Output files:")
        print(f"  - Mask: {consolidated_mask_path}")
        print(f"  - Bbox Image: {bbox_image_path}")
        print(f"  - JSON: {bbox_json_path}")
        if ground_truth:
            print(f"  - Metrics: Included in JSON (8 categories)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

