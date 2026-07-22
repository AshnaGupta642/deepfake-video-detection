import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

        # Remove classifier (FC layer)
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_features(image_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FeatureExtractor().to(device)
    model.eval()

    feature_vectors = []

    for file in os.listdir(image_folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(image_folder, file)

            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model(image)

            feature_vectors.append(feature.cpu())

    return feature_vectors