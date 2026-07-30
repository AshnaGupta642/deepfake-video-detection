import torch.nn as nn

from .attention import AttentionLayer
from .classifier import Classifier


class DeepFakeDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.attention = AttentionLayer()
        self.classifier = Classifier()

    def forward(self, x):

        # x shape:
        # (batch, 640)

        x = self.attention(x)
        x = self.classifier(x)

        return x