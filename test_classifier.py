import torch
from models.bilstm import BiLSTM
from models.classifier import Classifier


x = torch.randn(1,204,640)


bilstm = BiLSTM()
classifier = Classifier()


features = bilstm(x)

prediction = classifier(features)


print("Prediction Shape:", prediction.shape)
print("Prediction:", prediction)