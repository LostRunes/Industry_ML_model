import os
import gdown

MODEL_URLS = {
    "tep_fault_classifier.pkl": "https://drive.google.com/file/d/1JWlC_Hre2UAZG1borCOho1JZfAGbimVS/view?usp=drive_link",
    "anomaly_detector.pkl": "https://drive.google.com/file/d/1WundCySg8KcWQdHJRLXR8p7Ne3RfMl4o/view?usp=drive_link",
    "anomaly_scaler.pkl": "https://drive.google.com/file/d/1Lcbt4sD0_Qb5HLfZa3n1SOkB2fkNQceJ/view?usp=drive_link",
    "tep_scaler.pkl": "https://drive.google.com/file/d/1C-Al6_H_pu-ERtLtEJjusHGPykWgNhuO/view?usp=drive_link",
    "fault_classifier.pkl": "https://drive.google.com/file/d/1GnBE5pWPkMs4ukVotTfqk_SiS63dOtSE/view?usp=drive_link",
    "pca_model.pkl": "https://drive.google.com/file/d/1ADZvcjKOv7CT3LSv6Wh6d-lVEDZh3j3X/view?usp=drive_link"
}

os.makedirs("models", exist_ok=True)

for filename, file_id in MODEL_URLS.items():
    output_path = f"models/{filename}"

    if not os.path.exists(output_path):
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)