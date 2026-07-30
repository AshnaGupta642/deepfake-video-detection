import torch
from models.fusion import FeatureFusion

resnet = torch.load("features/ResNet_features.pt")
adfmpc = torch.load("features/ADFMPC_features.pt")

fusion = FeatureFusion()

output = fusion(resnet, adfmpc)

print("ResNet :", resnet.shape)
print("ADFMPC :", adfmpc.shape)
print("Fusion :", output.shape)