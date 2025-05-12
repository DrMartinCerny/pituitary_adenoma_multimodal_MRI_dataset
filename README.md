# Code Repository: Open-Access Multimodal MRI Dataset of Pituitary Adenoma

This repository accompanies the article:

**Černý, Valošek, Májovský, Sedlák et al.**  
*Open-access multimodal MRI dataset of pituitary adenoma*  
[Under review in *Scientific Data*]

All code written and maintained by **Martin Černý**.

If you use this dataset or code, please cite the corresponding article:

> Černý M, Valošek J, Májovský M, Sedlák V, et al. *An open-access multiparametric MRI dataset of pituitary adenomas with expert annotations and benchmark segmentation models*. *Scientific Data*. [Under review].

The dataset can be downloaded from:

[github.com/DrMartinCerny/pituitary_adenoma_multimodal_MRI_dataset](https://github.com/DrMartinCerny/pituitary_adenoma_multimodal_MRI_dataset)

The dataset provides a comprehensive, BIDS-organized collection of **multiparametric MRI scans** and **expert-validated segmentations** from 50 patients with pituitary adenomas. It is designed to support development and evaluation of segmentation models targeting challenging, pathologically altered sellar anatomy.

The dataset includes:
- Up to 10 MRI sequences per subject
- Ground truth annotations for 5 anatomical structures
- Outputs from 5 segmentation models
- Full benchmarking metrics

---

## Repository Contents

This repository contains all scripts used for:
- BIDS-compliant data conversion from raw DICOM
- Manual and automated segmentation processing
- Benchmark model evaluation (DSC, HD, IoU)
- Defacing of navigation scans

It is intended to support reproducible benchmarking and downstream ML workflows.

---

## Usage  

### 1. Set the Dataset Location  
Specify the dataset location in the `dataset_location.txt` file. The default location is:

```data/```

### 2. Evaluate Segmentation Accuracy
To evaluate the segmentation accuracy, use the following command:

```python dice.py <evaluated_segmentation_name> <label_mapping_file> [--keyframesOnly]```

*Example:*

```python dice.py predictionCerny2025 label_mapping/label_mapping_cerny_2025.json```

The label mapping file is required to account for different class labels in different model outputs. The `--keyframesOnly` flag will limit the evaluation extent to keyframes only.

---

## Dataset Creation

For reproducibility purposes, this repository contains scripts used to create this dataset

### 1. Convert the dataset from DICOM to NIfTI
This scripts requires that `source.tsv` is located in the previously specified dataset location and contains paths and series numbers (dicom tag 0020|0011) of series corresponding to respective imaging sequences.

```python convert_to_nifti.py```

Running this script will generate corresponding .nii.gz files and respective .json sidecars

### 2. Defacing
Facial features were removed from all 3D Navigation scans. In the previous step, empty defacing masks have been generated in `/derivatives/defaceMasks/`. Running the following script will apply them to the dataset:

```python deface.py```

---

### License

Code in this repository is licensed under the MIT License.
The dataset is licensed under CC BY-NC 4.0 – non-commercial use only, with attribution.