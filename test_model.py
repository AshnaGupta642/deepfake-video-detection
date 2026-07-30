import torch

from models.deepfake_model import DeepFakeDetector


model = DeepFakeDetector()

x = torch.randn(1,204,640)

output = model(x)

print("Final Output Shape:", output.shape)
print("Prediction:", output)