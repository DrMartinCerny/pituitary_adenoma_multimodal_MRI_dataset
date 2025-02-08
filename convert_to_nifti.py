import pandas as pd
import SimpleITK as sitk
import os
import json
import numpy as np

# Load dataset location
with open('dataset_location.txt', "r") as f:
    dataset_location = f.read().strip()

# Load a table with paths and Series Instance UIDs for all MRI data
participants = pd.read_csv(os.path.join(dataset_location, "source.tsv"), sep='\t')
print(participants)

# Function to extract only BIDS-compliant metadata
def extract_bids_metadata(dicom_reader, modality):
    bids_metadata = {
        "Modality": dicom_reader.GetMetaData("0008|0060").strip(),  # Modality
        "Manufacturer": dicom_reader.GetMetaData("0008|0070").strip(),  # Manufacturer
        "MagneticFieldStrength": dicom_reader.GetMetaData("0018|0087").strip(),  # Field Strength
        "RepetitionTime": dicom_reader.GetMetaData("0018|0080").strip(),  # TR
        "EchoTime": dicom_reader.GetMetaData("0018|0081").strip(),  # TE
        "FlipAngle": dicom_reader.GetMetaData("0018|1314").strip(),  # Flip Angle
        "InstitutionName": dicom_reader.GetMetaData("0008|0080").strip(),  # Institution Name
        "SeriesDescription": dicom_reader.GetMetaData("0008|103e").strip(),  # Series Description
    }
    
    # Additional DWI-specific metadata
    if modality == "dwi":
        bids_metadata.update({
            "DiffusionBValue": dicom_reader.GetMetaData("0018|9087").strip() if dicom_reader.HasMetaDataKey("0018|9087") else "N/A",  # Diffusion B-value
            "DiffusionDirectionality": dicom_reader.GetMetaData("0018|9075").strip() if dicom_reader.HasMetaDataKey("0018|9075") else "N/A",  # Diffusion Directionality
            "NumberOfGradientDirections": dicom_reader.GetMetaData("0018|9076").strip() if dicom_reader.HasMetaDataKey("0018|9076") else "N/A",  # Number of Gradient Directions
        })
    
    return bids_metadata

# Iterate over all participants and sequences
for _, participant in participants.iterrows():
    for sequence in participants.columns[2:]:  # Assuming first two columns are 'id' and 'path'
        if pd.notna(participant[sequence]):
            sub_id = f"sub-{participant['id']:03d}"
            modality = "dwi" if sequence in ["ADCb200_dwi", "ADCb1000_dwi", "eADCb1000_dwi"] else "anat"
            output_folder = os.path.join(dataset_location, sub_id, modality)
            filename = os.path.join(output_folder, f"{sub_id}_acq-{sequence}.nii.gz")
            json_filename = os.path.join(output_folder, f"{sub_id}_acq-{sequence}.json")
            os.makedirs(output_folder, exist_ok=True)
            
            # Read DICOM series
            reader = sitk.ImageSeriesReader()
            series_file_names = reader.GetGDCMSeriesFileNames(participant['path'], participant[sequence])
            reader.SetFileNames(series_file_names)
            image = reader.Execute()
            
            # Save image as NIfTI
            sitk.WriteImage(image, filename)
            
            # Extract BIDS-compliant metadata and save as JSON sidecar
            dicom_reader = sitk.ImageFileReader()
            dicom_reader.SetFileName(series_file_names[0])  # Read metadata from first file in series
            dicom_reader.LoadPrivateTagsOn()
            dicom_reader.ReadImageInformation()
            
            bids_metadata = extract_bids_metadata(dicom_reader, modality)
            
            # Write metadata to JSON
            with open(json_filename, 'w') as json_file:
                json.dump(bids_metadata, json_file, indent=4)

            # Create an empty segmentation mask for defacing if the sequence is CE3DNavigation_T1w
            if sequence == "CE3DNavigation_T1w":
                mask_folder = os.path.join(dataset_location, 'derivatives', 'defaceMasks', sub_id, 'anat')
                mask_filename = os.path.join(mask_folder, f"{sub_id}_acq-CE3DNavigation_desc-defacemask.nii.gz")
                os.makedirs(mask_folder, exist_ok=True)

                # Create an empty mask with the same shape and spatial properties as the original image
                mask_array = np.zeros(sitk.GetArrayFromImage(image).shape, dtype=np.uint8)
                mask_image = sitk.GetImageFromArray(mask_array)
                mask_image.CopyInformation(image)  # Ensure the mask has the same coordinate space
                
                # Save the mask
                sitk.WriteImage(mask_image, mask_filename)
            
            print(f"Processed: {sub_id}, {sequence}")