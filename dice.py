import json
import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import glob
import os
import sys

def calculate_dice_coefficient(gt, pred):
    if np.sum(gt) == 0:
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
    
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    
    gt_to_pred_distances, _ = pred_tree.query(gt_points)
    pred_to_gt_distances, _ = gt_tree.query(pred_points)
    
    hausdorff_distance = max(np.max(gt_to_pred_distances), np.max(pred_to_gt_distances))
    return hausdorff_distance

def calculate_iou(gt, pred):
    if np.sum(gt) == 0:
        return None
    intersection = np.sum(gt & pred)
    union = np.sum(gt | pred)
    iou = intersection / union if union > 0 else 1.0
    return iou

def main():
    if len(sys.argv) < 3:
        print("Usage: python dice.py <evaluated segmentation name> <label mapping file> [--keyframesOnly]")
        sys.exit(1)
    
    evaluated_segmentation = sys.argv[1]
    label_mapping_file = sys.argv[2]
    keyframes_only = '--keyframesOnly' in sys.argv

    with open('dataset_location.txt', "r") as f:
        dataset_location = f.read().strip()
    
    with open(label_mapping_file, "r") as f:
        label_mapping = json.load(f)

    tumor_metrics = {'subjectID': [], 'DSC': [], 'Hausdorff': [], 'IoU': []}
    label_metrics_data = {label['name']: {'dsc': [], 'hausdorff': [], 'iou': []} for label in label_mapping}
    subjects = sorted(glob.glob(os.path.join(dataset_location, 'derivatives', 'segmentations', '*')))
    print(f"There are {len(subjects)} found for evaluation")
    
    for subject in subjects:
        subjectID = subject.split('sub-')[-1]
        gt_path = os.path.join(subject, 'anat', f'sub-{subjectID}_label-groundTruth.nii.gz')
        pred_path = os.path.join(subject, 'anat', f'sub-{subjectID}_label-{evaluated_segmentation}.nii.gz')
        json_path = os.path.join(subject, 'anat', f'sub-{subjectID}_label-groundTruth.json')
        
        if os.path.exists(gt_path) and os.path.exists(pred_path):
            print(f"SUBJECT {subjectID}")
            ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
            prediction = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
            
            if keyframes_only and os.path.exists(json_path):
                with open(json_path, "r") as f:
                    json_sidecar = json.load(f)
                ground_truth = ground_truth[json_sidecar['KeyFrames']]
                prediction = prediction[json_sidecar['KeyFrames']]
            
            for label in label_mapping:
                gt_mask = np.isin(ground_truth, label['gt_labels'])
                pred_mask = np.isin(prediction, label['pred_labels'])
                
                dice_coefficient = calculate_dice_coefficient(gt_mask, pred_mask)
                hausdorff_distance = calculate_hausdorff_distance(gt_mask, pred_mask)
                iou = calculate_iou(gt_mask, pred_mask)
                
                label_metrics_data[label['name']]['dsc'].append(dice_coefficient if dice_coefficient is not None else np.nan)
                label_metrics_data[label['name']]['hausdorff'].append(hausdorff_distance if hausdorff_distance is not None else np.nan)
                label_metrics_data[label['name']]['iou'].append(iou if iou is not None else np.nan)
                
                if label['name'] == 'tumor':
                    tumor_metrics['subjectID'].append(subjectID)
                    tumor_metrics['DSC'].append(dice_coefficient if dice_coefficient is not None else np.nan)
                    tumor_metrics['Hausdorff'].append(hausdorff_distance if hausdorff_distance is not None else np.nan)
                    tumor_metrics['IoU'].append(iou if iou is not None else np.nan)
    
    print("\nPer-Label Metrics:")
    for label_name, data in label_metrics_data.items():
        if data['dsc']:
            avg_dsc = np.nanmean(data['dsc'])
            std_dsc = np.nanstd(data['dsc'])
        else:
            avg_dsc = None
            std_dsc = None
        
        if data['hausdorff']:
            avg_hausdorff = np.nanmean(data['hausdorff'])
            std_hausdorff = np.nanstd(data['hausdorff'])
        else:
            avg_hausdorff = None
            std_hausdorff = None
        
        if data['iou']:
            avg_iou = np.nanmean(data['iou'])
            std_iou = np.nanstd(data['iou'])
        else:
            avg_iou = None
            std_iou = None

        print(f" {label_name}: "
              f"Average DSC = {f'{avg_dsc:.3f} ± {std_dsc:.3f}' if avg_dsc is not None else 'N/A'}, "
              f"Average Hausdorff = {f'{avg_hausdorff:.3f} ± {std_hausdorff:.3f}' if avg_hausdorff is not None else 'N/A'}, "
              f"Average IoU = {f'{avg_iou:.3f} ± {std_iou:.3f}' if avg_iou is not None else 'N/A'}")

    # Save results to an Excel file
    output_filename = f"evaluated_segmentation_{'keyframes' if keyframes_only else 'full'}.xlsx"
    df = pd.DataFrame(tumor_metrics)
    df.to_excel(output_filename, index=False)
    print(f"Saved tumor evaluation metrics to {output_filename}")

if __name__ == "__main__":
    main()
