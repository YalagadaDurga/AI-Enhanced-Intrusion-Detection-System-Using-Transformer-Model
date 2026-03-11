import torch
import numpy as np
import pandas as pd
from transformer_model import TransformerIDS
from sklearn.metrics import accuracy_score, classification_report
import os

# -----------------------------
# Config
# -----------------------------
INPUT_DIM = 122
NUM_CLASSES = 5
LABEL_COLUMN = "label"   # this exists in processed CSV

label_map = {
    0: "Normal",
    1: "DoS",
    2: "Probe",
    3: "R2L",
    4: "U2R"
}

device = torch.device("cpu")

# -----------------------------
# Load model
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), "ids_transformer.pt")

model = TransformerIDS(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully")

# -----------------------------
# Load PROCESSED test CSV
# -----------------------------
csv_file = input("Enter labeled CSV file path: ").strip()
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"{csv_file} not found!")

df = pd.read_csv(csv_file)

# -----------------------------
# Split features & labels
# -----------------------------
y_true = df[LABEL_COLUMN].values
X = df.drop(columns=[LABEL_COLUMN]).values

if X.shape[1] != INPUT_DIM:
    raise ValueError(f"Expected {INPUT_DIM} features, got {X.shape[1]}")

X = torch.tensor(X, dtype=torch.float32)

# -----------------------------
# Predict
# -----------------------------
with torch.no_grad():
    outputs = model(X)
    y_pred = torch.argmax(outputs, dim=1).numpy()

# -----------------------------
# Metrics
# -----------------------------
acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 Test Accuracy: {acc:.4f}\n")

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_map.values()))
