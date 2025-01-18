# Code Repository for Open-Access Multimodal MRI Dataset of Pituitary Adenoma  

This repository accompanies the article:  

**Černý, Valošek, Májovský et al.**, *Open-access multimodal MRI dataset of pituitary adenoma*  

If you use this dataset, please cite the corresponding article:  
`{article citation}`  

---

## Usage  

### 1. Set the Dataset Location  
Specify the dataset location in the `dataset_location.txt` file. The default location is:

`data/`

### 2. Download the Dataset  
Run the following command to download the dataset to the previously defined dataset location:

`python download.py`

### 3. Evaluate Segmentation Accuracy
To evaluate the segmentation accuracy, use the following command:

`python script_name.py <evaluated_segmentation_name> <label_mapping_file>`

*Example:*

`python dice.py predictionCerny2025 label_mapping/label_mapping_cerny_2025.json`

The label mapping file is required to account for different class labels in different model outputs.
