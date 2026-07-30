import torch

print("SDFVD_FAKE:", torch.load("features/SDFVD_FAKE.pt").shape)
print("SDFVD_REAL:", torch.load("features/SDFVD_REAL.pt").shape)
print("UADFV_FAKE:", torch.load("features/UADFV_FAKE.pt").shape)
print("UADFV_REAL:", torch.load("features/UADFV_REAL.pt").shape)

print("ADFMPC:", torch.load("features/ADFMPC_features.pt").shape)