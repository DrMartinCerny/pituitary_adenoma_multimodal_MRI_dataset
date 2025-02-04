# Code Repository for Open-Access Multimodal MRI Dataset of Pituitary Adenoma  

This repository accompanies the article:  

**Černý, Valošek, Májovský, Sedlák et al.**, *Open-access multimodal MRI dataset of pituitary adenoma*  

All code written by Martin Černý

If you use this dataset, please cite the corresponding article:  
`{article citation}`  

---

## Usage  

### 1. Set the Dataset Location  
Specify the dataset location in the `dataset_location.txt` file. The default location is:

```data/```

### 2. Download the Dataset  
Run the following command to download the dataset to the previously defined dataset location:

```python download.py```

### 3. Evaluate Segmentation Accuracy
To evaluate the segmentation accuracy, use the following command:

```python dice.py <evaluated_segmentation_name> <label_mapping_file>```

*Example:*

```python dice.py predictionCerny2025 label_mapping/label_mapping_cerny_2025.json```

The label mapping file is required to account for different class labels in different model outputs.

---

## Dataset Creation

For reproducibility purposes, this repository contains scripts used to create this dataset

### 1. Convert the dataset from DICOM to NIfTI
This scripts requires that `source.tsv` is located in the previously specified dataset location and contains paths and series numbers (dicom tag 0020|0011) of series corresponding to respective imaging sequences.

```python convert_to_nifti.py```

Running this script will generate corresponding .nii.gz files and respective .json sidecars

### 2. Defacing
Facial features were removed from all 3D Navigation scans using `pydeface`. Following script will itterate over all subjects and deface them:

```python deface.py```