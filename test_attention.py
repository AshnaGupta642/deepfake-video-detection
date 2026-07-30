import torch

from models.fusion import FeatureFusion
from models.attention import AttentionLayer

# Load Features
resnet = torch.load("features/ResNet_features.pt")
adfmpc = torch.load("features/ADFMPC_features.pt")

# Fusion
fusion = FeatureFusion()
fused = fusion(resnet, adfmpc)

# Attention
attention = AttentionLayer()
output = attention(fused)

print("Fusion Shape :", fused.shape)
print("Attention Output :", output.shape)