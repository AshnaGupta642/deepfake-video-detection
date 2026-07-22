import torch
from feature_extractor import FeatureExtractor

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = FeatureExtractor().to(device)
model.eval()

# Dummy input (1 image, 3 channels, 112x112)
dummy = torch.randn(1, 3, 112, 112).to(device)

# Forward pass
with torch.no_grad():
    features = model(dummy)

print("Feature Vector Shape:", features.shape)
print(features)