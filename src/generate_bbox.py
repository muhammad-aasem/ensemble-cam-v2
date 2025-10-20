#!/usr/bin/env python3
"""
Generate tight bounding box from binary mask (P3).
Creates JSON with bounding box coordinates and visualization image.

Usage:
    python src/generate_bbox.py --input-mask path/to/mask_P3.png
    python src/generate_bbox.py --input-mask OUTPUT/00029391_000/densenet121_Cardiomegaly_P3.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def find_bounding_box(mask_path):
    """
    Find tight bounding box around non-zero regions in a binary mask.
    
    Args:
        mask_path: Path to binary mask image (P3)
    
    Returns:
        Dictionary with bounding box coordinates:
        {
            'x': int,      # Top-left x coordinate
            'y': int,      # Top-left y coordinate
            'width': int,  # Box width
            'height': int, # Box height
            'x2': int,     # Bottom-right x coordinate
            'y2': int      # Bottom-right y coordinate
        }
        Returns None if no non-zero regions found.
    """
    # Read mask image
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask image: {mask_path}")
    
    # Find all non-zero pixels
    coords = cv2.findNonZero(mask)
    
    if coords is None:
        print("Warning: No non-zero regions found in mask")
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


def draw_bounding_box(background_image, bbox, color=(0, 255, 0), thickness=3):
    """
    Draw bounding box on a background image.
    
    Args:
        background_image: Background image (numpy array) to draw on
        bbox: Bounding box dictionary from find_bounding_box()
        color: BGR color tuple (default: green)
        thickness: Line thickness in pixels (default: 3)
    
    Returns:
        Image with bounding box drawn
    """
    # Make a copy to avoid modifying the original
    img = background_image.copy()
    
    # Draw rectangle
    if bbox is not None:
        cv2.rectangle(
            img,
            (bbox['x'], bbox['y']),
            (bbox['x2'], bbox['y2']),
            color,
            thickness
        )
    
    return img


def draw_bounding_boxes(background_image, predicted_bbox, ground_truth_bbox=None, 
                        pred_color=(0, 255, 0), gt_color=(0, 0, 255), thickness=3):
    """
    Draw predicted and ground truth bounding boxes on a background image.
    
    Args:
        background_image: Background image (numpy array) to draw on
        predicted_bbox: Predicted bounding box dictionary
        ground_truth_bbox: Optional ground truth bounding box dictionary
        pred_color: BGR color for predicted bbox (default: green)
        gt_color: BGR color for ground truth bbox (default: red)
        thickness: Line thickness in pixels (default: 3)
    
    Returns:
        Image with bounding boxes drawn
    """
    # Make a copy to avoid modifying the original
    img = background_image.copy()
    
    # Draw ground truth first (so predicted is on top if overlapping)
    if ground_truth_bbox is not None:
        cv2.rectangle(
            img,
            (ground_truth_bbox['x'], ground_truth_bbox['y']),
            (ground_truth_bbox['x2'], ground_truth_bbox['y2']),
            gt_color,
            thickness
        )
        # Add label
        cv2.putText(
            img,
            'Ground Truth',
            (ground_truth_bbox['x'], ground_truth_bbox['y'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            gt_color,
            2
        )
    
    # Draw predicted bbox
    if predicted_bbox is not None:
        cv2.rectangle(
            img,
            (predicted_bbox['x'], predicted_bbox['y']),
            (predicted_bbox['x2'], predicted_bbox['y2']),
            pred_color,
            thickness
        )
        # Add label
        label_y = predicted_bbox['y'] - 10
        # If ground truth exists and labels would overlap, offset the predicted label
        if ground_truth_bbox is not None and abs(predicted_bbox['y'] - ground_truth_bbox['y']) < 30:
            label_y = predicted_bbox['y'] - 40
        cv2.putText(
            img,
            'Predicted',
            (predicted_bbox['x'], label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            pred_color,
            2
        )
    
    return img


def parse_ground_truth(gt_string):
    """
    Parse ground truth CSV string.
    
    Args:
        gt_string: CSV string in format "Label,x,y,w,h" (e.g., "Cardiomegaly,277,459,540,301")
    
    Returns:
        Dictionary with ground truth bounding box coordinates
    """
    try:
        parts = gt_string.split(',')
        if len(parts) != 5:
            raise ValueError(f"Expected 5 values (Label,x,y,w,h), got {len(parts)}")
        
        label = parts[0].strip()
        x = int(parts[1].strip())
        y = int(parts[2].strip())
        w = int(parts[3].strip())
        h = int(parts[4].strip())
        
        return {
            'label': label,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'x2': x + w,
            'y2': y + h
        }
    except Exception as e:
        raise ValueError(f"Failed to parse ground truth string '{gt_string}': {e}")


def compute_iou(bbox1, bbox2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box dict with keys: x, y, width, height
        bbox2: Second bounding box dict with keys: x, y, width, height
    
    Returns:
        IoU score (float between 0 and 1)
    """
    # Calculate intersection
    x_left = max(bbox1['x'], bbox2['x'])
    y_top = max(bbox1['y'], bbox2['y'])
    x_right = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
    y_bottom = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    bbox1_area = bbox1['width'] * bbox1['height']
    bbox2_area = bbox2['width'] * bbox2['height']
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    iou = intersection_area / union_area
    return float(iou)


def compute_dice_coefficient(bbox1, bbox2):
    """
    Compute Dice coefficient between two bounding boxes.
    
    Args:
        bbox1: First bounding box dict
        bbox2: Second bounding box dict
    
    Returns:
        Dice coefficient (float between 0 and 1)
    """
    # Calculate intersection
    x_left = max(bbox1['x'], bbox2['x'])
    y_top = max(bbox1['y'], bbox2['y'])
    x_right = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
    y_bottom = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas
    bbox1_area = bbox1['width'] * bbox1['height']
    bbox2_area = bbox2['width'] * bbox2['height']
    
    # Dice = 2 * |A ∩ B| / (|A| + |B|)
    dice = (2.0 * intersection_area) / (bbox1_area + bbox2_area)
    return float(dice)


def compute_pixel_accuracy(bbox1, bbox2, image_shape):
    """
    Compute pixel-level accuracy between two bounding boxes.
    
    Args:
        bbox1: First bounding box dict
        bbox2: Second bounding box dict
        image_shape: Tuple of (height, width)
    
    Returns:
        Pixel accuracy (float between 0 and 1)
    """
    total_pixels = image_shape[0] * image_shape[1]
    
    # Calculate intersection (True Positives)
    x_left = max(bbox1['x'], bbox2['x'])
    y_top = max(bbox1['y'], bbox2['y'])
    x_right = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
    y_bottom = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])
    
    if x_right < x_left or y_bottom < y_top:
        tp = 0
    else:
        tp = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate areas
    bbox1_area = bbox1['width'] * bbox1['height']
    bbox2_area = bbox2['width'] * bbox2['height']
    
    # False Positives: pixels in bbox1 but not in intersection
    fp = bbox1_area - tp
    
    # False Negatives: pixels in bbox2 but not in intersection
    fn = bbox2_area - tp
    
    # True Negatives: pixels outside both bboxes
    tn = total_pixels - bbox1_area - bbox2_area + tp
    
    # Pixel Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / total_pixels
    return float(accuracy)


def compute_colloc(iou, threshold):
    """
    Compute Correct Localization (ColLoc) metric.
    ColLoc indicates whether the IoU meets or exceeds a threshold.
    
    Args:
        iou: IoU score between predicted and ground truth
        threshold: IoU threshold for correct localization
    
    Returns:
        1.0 if IoU >= threshold, 0.0 otherwise
    """
    return 1.0 if iou >= threshold else 0.0


def compute_precision_recall(bbox_pred, bbox_gt, iou_threshold=0.5):
    """
    Compute precision and recall for object detection.
    
    Args:
        bbox_pred: Predicted bounding box
        bbox_gt: Ground truth bounding box
        iou_threshold: IoU threshold to consider detection as correct
    
    Returns:
        Tuple of (precision, recall)
    """
    iou = compute_iou(bbox_pred, bbox_gt)
    
    # For single box comparison:
    # - If IoU >= threshold: True Positive
    # - If IoU < threshold: False Positive (pred) and False Negative (gt)
    
    if iou >= iou_threshold:
        # True Positive
        precision = 1.0
        recall = 1.0
    else:
        # False Positive and False Negative
        precision = 0.0
        recall = 0.0
    
    return precision, recall


def compute_map(bbox_pred, bbox_gt):
    """
    Compute mean Average Precision (mAP) at different IoU thresholds.
    For single bbox comparison, this computes precision at IoU thresholds [0.5, 0.55, ..., 0.95].
    
    Args:
        bbox_pred: Predicted bounding box
        bbox_gt: Ground truth bounding box
    
    Returns:
        Dictionary with mAP scores
    """
    iou = compute_iou(bbox_pred, bbox_gt)
    
    # IoU thresholds from 0.5 to 0.95 in steps of 0.05 (COCO style)
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    
    precisions = []
    for threshold in thresholds:
        if iou >= threshold:
            precisions.append(1.0)
        else:
            precisions.append(0.0)
    
    map_score = sum(precisions) / len(precisions)
    
    return {
        'mAP@[.5:.95]': float(map_score),
        'AP@.5': float(precisions[0]),
        'AP@.75': float(precisions[5]) if len(precisions) > 5 else 0.0
    }


def save_bbox_json(bbox, output_path, mask_path, image_shape, ground_truth=None, metrics=None):
    """
    Save bounding box information to JSON file.
    
    Args:
        bbox: Bounding box dictionary
        output_path: Path to save JSON file
        mask_path: Original mask file path (for metadata)
        image_shape: Image dimensions (height, width)
        ground_truth: Optional ground truth bounding box dictionary
        metrics: Optional metrics dictionary
    """
    data = {
        'bounding_box': bbox,
        'image_dimensions': {
            'height': int(image_shape[0]),
            'width': int(image_shape[1])
        },
        'source_mask': str(mask_path),
        'format': 'x, y, width, height'
    }
    
    if ground_truth is not None:
        data['ground_truth'] = ground_truth
    
    if metrics is not None:
        data['metrics'] = metrics
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    """Main function for bounding box generation."""
    parser = argparse.ArgumentParser(
        description="Generate tight bounding box from binary mask (P3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate bounding box from P3 mask
  python src/generate_bbox.py --input-mask OUTPUT/00029391_000/densenet121_Cardiomegaly_P3.png
  
  # With custom output directory
  python src/generate_bbox.py --input-mask path/to/mask_P3.png --output-dir custom_output/
  
  # Custom bounding box color and thickness
  python src/generate_bbox.py --input-mask mask_P3.png --color 255 0 0 --thickness 5
  
  # With ground truth for metric computation (draws both predicted & ground truth bboxes)
  python src/generate_bbox.py --input-mask mask_P3.png --ground-truth "Cardiomegaly,277,459,540,301"

Output files:
  {model}_{pathology}_PBBox.json - Bounding box coordinates (+ ground truth & metrics if provided)
  {model}_{pathology}_PBBox.png  - Visualization (predicted=green, ground truth=red if provided)
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--input-mask',
        type=str,
        required=True,
        help='Path to input binary mask (P3.png)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as input mask)'
    )
    
    parser.add_argument(
        '--color',
        type=int,
        nargs=3,
        default=[0, 255, 0],
        metavar=('B', 'G', 'R'),
        help='Bounding box color in BGR format (default: 0 255 0 = green)'
    )
    
    parser.add_argument(
        '--thickness',
        type=int,
        default=3,
        help='Bounding box line thickness in pixels (default: 3)'
    )
    
    parser.add_argument(
        '--ground-truth',
        type=str,
        default=None,
        help='Ground truth bounding box as CSV string: "Label,x,y,w,h" (e.g., "Cardiomegaly,277,459,540,301")'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_mask)
    if not input_path.exists():
        print(f"Error: Input mask not found: {args.input_mask}", file=sys.stderr)
        sys.exit(1)
    
    if not input_path.name.endswith('_P3.png'):
        print(f"Warning: Input file does not end with '_P3.png'. Expected binary mask.")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent
    
    # Generate output filenames by replacing _P3.png with _PBBox
    base_name = input_path.stem.replace('_P3', '')
    json_output = output_dir / f"{base_name}_PBBox.json"
    image_output = output_dir / f"{base_name}_PBBox.png"
    
    print(f"Input mask: {input_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Find bounding box
        print("\nFinding bounding box...")
        bbox = find_bounding_box(input_path)
        
        if bbox is None:
            print("❌ Error: No bounding box could be generated (empty mask)")
            sys.exit(1)
        
        print(f"Bounding box found:")
        print(f"  Position: ({bbox['x']}, {bbox['y']})")
        print(f"  Size: {bbox['width']} × {bbox['height']} pixels")
        print(f"  Bottom-right: ({bbox['x2']}, {bbox['y2']})")
        
        # Get image dimensions from mask
        mask = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        image_shape = mask.shape
        
        # Parse ground truth and compute metrics if provided
        ground_truth = None
        metrics = None
        
        if args.ground_truth:
            print(f"\nParsing ground truth...")
            try:
                ground_truth = parse_ground_truth(args.ground_truth)
                print(f"Ground truth bbox:")
                print(f"  Label: {ground_truth['label']}")
                print(f"  Position: ({ground_truth['x']}, {ground_truth['y']})")
                print(f"  Size: {ground_truth['width']} × {ground_truth['height']} pixels")
                print(f"  Bottom-right: ({ground_truth['x2']}, {ground_truth['y2']})")
                
                # Compute metrics
                print(f"\nComputing metrics...")
                iou = compute_iou(bbox, ground_truth)
                dice = compute_dice_coefficient(bbox, ground_truth)
                pixel_acc = compute_pixel_accuracy(bbox, ground_truth, image_shape)
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
        
        # Save JSON
        print(f"\nSaving bounding box data: {json_output.name}")
        save_bbox_json(bbox, json_output, input_path, image_shape, ground_truth, metrics)
        
        # Load background image (_input_image_.png)
        input_image_path = output_dir / "_input_image_.png"
        if input_image_path.exists():
            print(f"Loading background image: {input_image_path.name}")
            background_image = cv2.imread(str(input_image_path))
            if background_image is None:
                print(f"  ⚠️  Warning: Could not read background image, using white canvas")
                background_image = np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8) * 255
        else:
            print(f"  ⚠️  Warning: Background image not found at {input_image_path}, using white canvas")
            background_image = np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8) * 255
        
        # Draw and save visualization
        print(f"Generating visualization: {image_output.name}")
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
                thickness=args.thickness
            )
        cv2.imwrite(str(image_output), viz_image)
        
        print(f"\n✅ Bounding box generation complete!")
        print(f"   JSON: {json_output}")
        print(f"   Image: {image_output}")
        if ground_truth:
            print(f"   Ground truth: Included with {len(metrics)} metric categories")
        
    except Exception as e:
        print(f"\n❌ Error during bounding box generation: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

