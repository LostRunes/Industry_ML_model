import os
import gdown

DATASET_URLS = {
    "tep_temporal_features.csv": "https://drive.google.com/file/d/1-iNc8QXjQo29-Az-5pkPmhskMt6J7AzL/view?usp=drive_link"
    
}

os.makedirs("data", exist_ok=True)

for filename, file_id in DATASET_URLS.items():
    output_path = f"data/{filename}"

    if not os.path.exists(output_path):
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)