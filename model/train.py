import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import os

from transformer_model import TransformerIDS

# ----------------------------
# Load Data
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "../data/processed")

X_train = np.load(os.path.join(data_dir, "X_train.npy"))
X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
y_train = np.load(os.path.join(data_dir, "y_train.npy"))
y_test  = np.load(os.path.join(data_dir, "y_test.npy"))

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ----------------------------
# Convert to Tensors
# ----------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

# ----------------------------
# Dataset & Loader
# ----------------------------
batch_size = 128

train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# ----------------------------
# Setup
# ----------------------------
input_dim = X_train.shape[1]
num_classes = len(torch.unique(y_train))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = TransformerIDS(
    input_dim=input_dim,
    num_classes=num_classes,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1
).to(device)

# ----------------------------
# 🔥 Handle Class Imbalance
# ----------------------------
classes = np.unique(y_train.numpy())

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.numpy()
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class Weights:", class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Training Loop
# ----------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# ----------------------------
# Evaluation
# ----------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print("\nTest Accuracy:", acc)

print("\nClassification Report:\n")
print(classification_report(
    all_labels,
    all_preds,
    zero_division=0   # avoids warning
))

# ----------------------------
# Save Model
# ----------------------------
model_path = os.path.join(BASE_DIR, "ids_transformer.pt")
torch.save(model.state_dict(), model_path)
print("\nModel saved to:", model_path)