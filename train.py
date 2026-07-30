import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

from models.fusion import FeatureFusion
from models.deepfake_model import DeepFakeDetector

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# Load Features
# -----------------------------
resnet_features = torch.load("features/ResNet_features.pt").float()
adfmpc_features = torch.load("features/ADFMPC_features.pt").float()
labels = torch.load("features/ResNet_labels.pt").float().unsqueeze(1)

print("ResNet:", resnet_features.shape)
print("ADFMPC:", adfmpc_features.shape)
print("Labels:", labels.shape)

# -----------------------------
# Train/Test Split
# -----------------------------
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    resnet_features,
    adfmpc_features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# -----------------------------
# Move Data to Device
# -----------------------------
X1_train = X1_train.to(device)
X2_train = X2_train.to(device)
y_train = y_train.to(device)

X1_test = X1_test.to(device)
X2_test = X2_test.to(device)
y_test = y_test.to(device)

# -----------------------------
# Models
# -----------------------------
fusion = FeatureFusion().to(device)
model = DeepFakeDetector().to(device)

criterion = nn.BCELoss()

optimizer = optim.Adam(
    list(fusion.parameters()) +
    list(model.parameters()),
    lr=0.001
)

print("Everything Loaded Successfully!")

# -----------------------------
# Training
# -----------------------------
epochs = 30

best_val_acc = 0.0
patience = 5
counter = 0

for epoch in range(epochs):

    # -------------------------
    # Train
    # -------------------------
    fusion.train()
    model.train()

    optimizer.zero_grad()

    fused = fusion(X1_train, X2_train)

    outputs = model(fused)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    preds = (outputs >= 0.5).float()

    train_acc = (preds == y_train).float().mean()

    # -------------------------
    # Validation
    # -------------------------
    fusion.eval()
    model.eval()

    with torch.no_grad():

        fused_test = fusion(X1_test, X2_test)

        test_outputs = model(fused_test)

        test_loss = criterion(test_outputs, y_test)

        test_preds = (test_outputs >= 0.5).float()

        test_acc = (test_preds == y_test).float().mean()

    # -------------------------
    # Save Best Model
    # -------------------------
    if test_acc.item() > best_val_acc:

        best_val_acc = test_acc.item()

        counter = 0

        torch.save(
            {
                "fusion": fusion.state_dict(),
                "model": model.state_dict(),
            },
            "best_model.pth",
        )

        print("✅ Best Model Saved!")

    else:

        counter += 1

    # -------------------------
    # Print Progress
    # -------------------------
    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {loss.item():.4f} | "
        f"Train Acc: {train_acc.item()*100:.2f}% | "
        f"Val Loss: {test_loss.item():.4f} | "
        f"Val Acc: {test_acc.item()*100:.2f}%"
    )

    # -------------------------
    # Early Stopping
    # -------------------------
    if counter >= patience:

        print("\nEarly Stopping Triggered!")

        break

print("\nTraining Completed!")
print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
print("Best model saved as: best_model.pth")