import glob
import SimpleITK as sitk
import os
import numpy as np

# Load dataset location
with open('dataset_location.txt', "r") as f:
    dataset_location = f.read().strip()

# Iterate over all anatomical images
for image_path in glob.glob(os.path.join(dataset_location, 'sub-*', 'anat', 'sub-*_acq-CE3DNavigation_T1w.nii.gz')):
    print(f"Processing: {image_path}")

    # Construct the expected defacing mask path
    sub_id = os.path.basename(image_path).split("_")[0]  # Extract subject ID
    mask_path = os.path.join(dataset_location, 'derivatives', 'defaceMasks', sub_id, 'anat', f"{sub_id}_acq-CE3DNavigation_desc-defacemask.nii.gz")

    # Check if the mask exists
    if not os.path.exists(mask_path):
        print(f"Skipping {image_path} (No defacing mask found)")
        continue

    # Load the anatomical image
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    # Load the defacing mask
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)

    # Ensure the mask and image have the same shape
    if mask_array.shape != image_array.shape:
        print(f"Skipping {image_path} (Mask and image shape mismatch)")
        continue

    # Apply the mask (set masked regions to 0)
    defaced_array = np.where(mask_array > 0, 0, image_array)

    # Convert back to SimpleITK image
    defaced_image = sitk.GetImageFromArray(defaced_array)
    defaced_image.CopyInformation(image)  # Preserve spatial metadata

    # Save the defaced image, overwriting the original
    sitk.WriteImage(defaced_image, image_path)

    print(f"Defaced and saved: {image_path}")
