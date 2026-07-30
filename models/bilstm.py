import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=640,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        return output