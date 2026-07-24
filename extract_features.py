import os
import torch
from PIL import Image
from feature_extractor import FeatureExtractor, transform

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FeatureExtractor().to(device)
model.eval()

# Dataset folder
DATASET_DIR = "../faces"

# Output folder
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with torch.no_grad():
    for category in os.listdir(DATASET_DIR):
        category_path = os.path.join(DATASET_DIR, category)

        if not os.path.isdir(category_path):
            continue

        category_features = []

        for video_folder in os.listdir(category_path):
            video_path = os.path.join(category_path, video_folder)

            if not os.path.isdir(video_path):
                continue

            for img_name in os.listdir(video_path):
                if img_name.endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(video_path, img_name)

                    image = Image.open(img_path).convert("RGB")
                    image = transform(image).unsqueeze(0).to(device)

                    feature = model(image)
                    category_features.append(feature.cpu())

        if len(category_features) > 0:
            category_features = torch.cat(category_features, dim=0)
            torch.save(
                category_features,
                os.path.join(OUTPUT_DIR, f"{category}.pt")
            )

print("Feature extraction completed!") 
