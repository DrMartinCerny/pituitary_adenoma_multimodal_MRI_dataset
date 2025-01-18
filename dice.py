import json
import SimpleITK as sitk
import numpy as np
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

def main():
    if len(sys.argv) < 3:
        print("Usage: python script_name.py <evaluated segmentation name> <label mapping file>")
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
    count_valid_dcs = 0  # Counter for valid DCS calculations

    # Dictionary to store per-label total and count
    label_dcs_data = {label['name']: {'total_dcs': 0, 'count': 0} for label in label_mapping}

    # Iterate over all subjects in the dataset
    for subject in glob.glob(os.path.join(dataset_location, 'derivatives', 'segmentation', '*')):
        subjectID = subject.split('sub-')[-1]
        print(f"SUBJECT {subjectID}")

        # Load ground truth and prediction images
        ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subject, 'anat', f'sub-{subjectID}-label_groundTruth.nii')))
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subject, 'anat', f'sub-{subjectID}-label_{evaluated_segmentation}.nii')))

        # Calculate DCS for each label in the mapping
        for label in label_mapping:
            dice_coefficient = calculate_dice_coefficient(np.isin(ground_truth, label['gt_labels']), np.isin(prediction, label['pred_labels']))
            if dice_coefficient is not None:
                count_valid_dcs += 1  # Increment valid DCS counter
                # Update per-label data
                label_dcs_data[label['name']]['total_dcs'] += dice_coefficient
                label_dcs_data[label['name']]['count'] += 1
            print(f" {label['name']}: DCS = {dice_coefficient}")

    # Display per-label DCS data
    print("\nPer-Label DCS:")
    for label_name, data in label_dcs_data.items():
        label_average_dcs = data['total_dcs'] / data['count'] if data['count'] > 0 else None
        print(f" {label_name}: Average DCS = {label_average_dcs}")

if __name__ == "__main__":
    main()