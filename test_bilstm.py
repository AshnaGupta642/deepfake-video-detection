import torch
from models.bilstm import BiLSTM


# fake attention output
x = torch.randn(1,204,640)

model = BiLSTM()

out = model(x)

print("BiLSTM Output:", out.shape)