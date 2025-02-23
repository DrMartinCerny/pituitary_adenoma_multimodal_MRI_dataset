import json
import SimpleITK as sitk
import numpy as np
from scipy.spatial import cKDTree
import glob
import os
import sys

def calculate_dice_coefficient(gt, pred):
    if np.sum(gt) == 0:
        # Skip evaluation if the label does not exist in ground truth
        return None
    intersection = np.sum(gt & pred)
    union = np.sum(gt) + np.sum(pred)
    dice_coefficient = (2.0 * intersection) / union if union > 0 else 1.0
    return dice_coefficient

def calculate_hausdorff_distance(gt, pred):
    if np.sum(gt) == 0 or np.sum(pred) == 0:
        return None

    gt_points = np.argwhere(gt)
    pred_points = np.argwhere(pred)

    if gt_points.shape[0] == 0 or pred_points.shape[0] == 0:
        return None

    # Build KD-trees
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)

    # Compute Hausdorff distances using nearest neighbor search
    gt_to_pred_distances, _ = pred_tree.query(gt_points)
    pred_to_gt_distances, _ = gt_tree.query(pred_points)

    hausdorff_distance = max(np.max(gt_to_pred_distances), np.max(pred_to_gt_distances))
    return hausdorff_distance

def calculate_iou(gt, pred):
    if np.sum(gt) == 0:
        # Skip evaluation if the label does not exist in ground truth
        return None
    intersection = np.sum(gt & pred)
    union = np.sum(gt | pred)
    iou = intersection / union if union > 0 else 1.0
    return iou

def main():
    if len(sys.argv) < 3:
        print("Usage: python dice.py <evaluated segmentation name> <label mapping file>")
        sys.exit(1)

    evaluated_segmentation = sys.argv[1]
    label_mapping_file = sys.argv[2]

    # Load dataset location
    with open('dataset_location.txt', "r") as f:
        dataset_location = f.read().strip()

    # Load label mapping
    with open(label_mapping_file, "r") as f:
        label_mapping = json.load(f)

    # Initialize variables
    count_valid_dsc = 0  # Counter for valid calculations

    # Dictionary to store per-label metrics data
    label_metrics_data = {label['name']: {'total_dsc': 0, 'total_hausdorff': 0, 'total_iou': 0, 'count': 0} for label in label_mapping}

    # Find all subjects for evaluation
    subjects = sorted(glob.glob(os.path.join(dataset_location, 'derivatives', 'segmentations', '*')))
    print(f"There are {len(subjects)} found for evaluation")
    
    # Iterate over all subjects in the dataset
    for subject in subjects:
        subjectID = subject.split('sub-')[-1]
        print(f"SUBJECT {subjectID}")

        # Load ground truth and prediction images
        ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subject, 'anat', f'sub-{subjectID}_label-groundTruth.nii.gz')))
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subject, 'anat', f'sub-{subjectID}_label-{evaluated_segmentation}.nii.gz')))

        # Calculate metrics for each label in the mapping
        for label in label_mapping:
            gt_mask = np.isin(ground_truth, label['gt_labels'])
            pred_mask = np.isin(prediction, label['pred_labels'])

            dice_coefficient = calculate_dice_coefficient(gt_mask, pred_mask)
            hausdorff_distance = calculate_hausdorff_distance(gt_mask, pred_mask)
            iou = calculate_iou(gt_mask, pred_mask)

            if dice_coefficient is not None:
                count_valid_dsc += 1
                # Update per-label data
                label_metrics_data[label['name']]['total_dsc'] += dice_coefficient
                label_metrics_data[label['name']]['total_hausdorff'] += hausdorff_distance if hausdorff_distance is not None else 0
                label_metrics_data[label['name']]['total_iou'] += iou if iou is not None else 0
                label_metrics_data[label['name']]['count'] += 1

            print(f" {label['name']}: DSC = {f'{dice_coefficient:.4f}' if dice_coefficient is not None else 'N/A'}, "
                f"Hausdorff = {f'{hausdorff_distance:.4f}' if hausdorff_distance is not None else 'N/A'}, "
                f"IoU = {f'{iou:.4f}' if iou is not None else 'N/A'}")

    # Display per-label metrics data
    print("\nPer-Label Metrics:")
    for label_name, data in label_metrics_data.items():
        label_average_dsc = data['total_dsc'] / data['count'] if data['count'] > 0 else None
        label_average_hausdorff = data['total_hausdorff'] / data['count'] if data['count'] > 0 else None
        label_average_iou = data['total_iou'] / data['count'] if data['count'] > 0 else None

        print(f" {label_name}: Average DSC = {f'{label_average_dsc:.4f}' if label_average_dsc is not None else 'N/A'}, "
            f"Average Hausdorff = {f'{label_average_hausdorff:.4f}' if label_average_hausdorff is not None else 'N/A'}, "
            f"Average IoU = {f'{label_average_iou:.4f}' if label_average_iou is not None else 'N/A'}")

if __name__ == "__main__":
    main()