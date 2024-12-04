import os
import requests
import zipfile
import shutil
from tqdm import tqdm

def download_with_progress(url, temp_file):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading")
    
    with open(temp_file, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        raise ValueError("Error occurred during download. File size mismatch.")
    print("Download complete.")

def unzip_file(temp_file, extract_to):
    """Unzip the downloaded file."""
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}.")

def main():
    # Read the dataset location
    dataset_location_file = 'dataset_location.txt'
    try:
        with open(dataset_location_file, 'r') as file:
            dataset_location = file.read().strip()
    except FileNotFoundError:
        print(f"Error: {dataset_location_file} not found.")
        return
    
    if not dataset_location:
        print("Error: Dataset location is empty in the file.")
        return
    
    dataset_folder = os.path.abspath(dataset_location)
    dataset_url = "https://storage.cloud.google.com/pituitary_dataset/pituitary_dataset.zip"  # Replace with your URL
    temp_zip = "temp_dataset.zip"
    
    # Check if the dataset folder exists
    if os.path.exists(dataset_folder):
        print(f"Dataset folder '{dataset_folder}' already exists.")
        choice = input("Do you want to continue and overwrite it? (yes/no): ").strip().lower()
        if choice != 'yes':
            print("Aborting.")
            return
        print("Removing existing folder...")
        shutil.rmtree(dataset_folder)
    
    try:
        # Download the dataset
        download_with_progress(dataset_url, temp_zip)
        
        # Unzip the dataset
        unzip_file(temp_zip, dataset_folder)
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
            print(f"Temporary file '{temp_zip}' removed.")

if __name__ == "__main__":
    main()
