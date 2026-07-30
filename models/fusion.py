import torch
import torch.nn as nn


class FeatureFusion(nn.Module):

    def __init__(self):
        super().__init__()

        # Project both features to same dimension
        self.resnet_fc = nn.Linear(2048, 512)
        self.adfmpc_fc = nn.Linear(97, 512)

        # Learnable fusion gate
        self.gate = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )

        # Final projection
        self.output_fc = nn.Linear(512, 640)

    def forward(self, resnet_features, adfmpc_features):

        resnet = self.resnet_fc(resnet_features)
        adfmpc = self.adfmpc_fc(adfmpc_features)

        # Learn adaptive weights
        gate = self.gate(torch.cat([resnet, adfmpc], dim=1))

        # Adaptive fusion
        fused = gate * resnet + (1 - gate) * adfmpc

        # Final feature vector
        fused = self.output_fc(fused)

        return fused