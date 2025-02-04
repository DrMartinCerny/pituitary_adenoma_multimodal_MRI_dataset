import os
import glob
import subprocess

# Read dataset location from file
with open("dataset_location.txt", "r") as f:
    base_path = f.read().strip()

# Construct the search pattern
search_pattern = os.path.join(base_path, "sub-*", "anat", "sub-*_acq-CEAxNavi_T1w.nii.gz")

# Find all matching files
nii_files = glob.glob(search_pattern)

# Loop through each file and deface in place
for nii_file in nii_files:
    print(f"Defacing: {nii_file}")
    try:
        subprocess.run(["pydeface", nii_file, "--cost", "corratio", "--force", "--applyto", nii_file], check=True)
        print(f"Successfully defaced: {nii_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error defacing {nii_file}: {e}")

print("Batch defacing complete!")
