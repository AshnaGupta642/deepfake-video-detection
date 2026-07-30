import os
import torch
from PIL import Image
from feature_extractor import FeatureExtractor, transform

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FeatureExtractor().to(device)
model.eval()

DATASET_DIR = "faces"

OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Final lists
all_features = []
all_labels = []
all_video_names = []

# Dataset labels
datasets = [
    ("SDFVD_FAKE", 1),
    ("SDFVD_REAL", 0),
    ("UADFV_FAKE", 1),
    ("UADFV_REAL", 0),
    ("CELEBDF_FAKE", 1),
    ("CELEBDF_REAL", 0)
]
with torch.no_grad():

    for category, label in datasets:

        category_path = os.path.join(DATASET_DIR, category)

        if not os.path.isdir(category_path):
            continue

        print(f"\nProcessing {category}")

        for video_folder in sorted(os.listdir(category_path)):
            print(f"Processing video: {video_folder}")


            video_path = os.path.join(category_path, video_folder)

            if not os.path.isdir(video_path):
                continue

            frame_features = []

            for img_name in sorted(os.listdir(video_path)):

                if img_name.endswith((".jpg", ".png", ".jpeg")):

                    img_path = os.path.join(video_path, img_name)

                    image = Image.open(img_path).convert("RGB")
                    image = transform(image).unsqueeze(0).to(device)

                    feature = model(image)

                    frame_features.append(feature.cpu())

            if len(frame_features) == 0:
                continue

            frame_features = torch.cat(frame_features, dim=0)

            # Average of all frame features
            video_feature = frame_features.mean(dim=0)

            all_features.append(video_feature)
            all_labels.append(label)
            all_video_names.append(video_folder)

features = torch.stack(all_features)

labels = torch.tensor(all_labels)

torch.save(features, "features/ResNet_features.pt")
torch.save(labels, "features/ResNet_labels.pt")
torch.save(all_video_names, "features/ResNet_video_names.pt")

print("\nExtraction Finished")
print("Videos :", len(all_video_names))
print("Feature Shape :", features.shape)
print("Labels Shape :", labels.shape)