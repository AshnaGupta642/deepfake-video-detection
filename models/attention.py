import torch
import torch.nn as nn


class AttentionLayer(nn.Module):

    def __init__(self, input_dim=640):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):

        attention_weights = self.attention(x)

        output = x * attention_weights

        return output
    