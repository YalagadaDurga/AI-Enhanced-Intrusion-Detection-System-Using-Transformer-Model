import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "kdd_train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "kdd_test.csv")

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# -----------------------------
# Label mapping (5 classes)
# -----------------------------
attack_mapping = {
    "normal": "Normal",

    # DoS
    "neptune": "DoS", "smurf": "DoS", "back": "DoS", "teardrop": "DoS",
    "pod": "DoS", "land": "DoS",

    # Probe
    "satan": "Probe", "ipsweep": "Probe", "portsweep": "Probe", "nmap": "Probe",

    # R2L
    "guess_passwd": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "spy": "R2L",

    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R",
    "rootkit": "U2R", "perl": "U2R"
}

# Target column is last column
TARGET_COL = train_df.columns[-1]

train_df["label"] = train_df[TARGET_COL].map(attack_mapping)
test_df["label"]  = test_df[TARGET_COL].map(attack_mapping)

train_df.dropna(subset=["label"], inplace=True)
test_df.dropna(subset=["label"], inplace=True)

class_encoding = {"Normal":0, "DoS":1, "Probe":2, "R2L":3, "U2R":4}

train_df["label"] = train_df["label"].map(class_encoding)
test_df["label"]  = test_df["label"].map(class_encoding)

train_df.drop(columns=[TARGET_COL], inplace=True)
test_df.drop(columns=[TARGET_COL], inplace=True)

# -----------------------------
# Feature split
# -----------------------------
categorical_cols = ["protocol_type", "service", "flag"]
numerical_cols = [c for c in train_df.columns if c not in categorical_cols + ["label"]]

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

# -----------------------------
# Encoding + Scaling
# -----------------------------
encoder = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

X_train_enc = encoder.fit_transform(X_train)
X_test_enc  = encoder.transform(X_test)

# -----------------------------
# Save encoder & scaler
# -----------------------------
joblib.dump(encoder, os.path.join(PROCESSED_DIR, "encoder.pkl"))

# -----------------------------
# Save numpy arrays
# -----------------------------
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train_enc)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train.values)

np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test_enc)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test.values)

# -----------------------------
# Save CSV for prediction
# -----------------------------
test_processed = pd.DataFrame(X_test_enc)
test_processed["label"] = y_test.values

test_processed.to_csv(
    os.path.join(PROCESSED_DIR, "kdd_test_processed.csv"),
    index=False
)

# -----------------------------
# Done
# -----------------------------
print("✅ Preprocessing completed")
print("X_train shape:", X_train_enc.shape)
print("X_test shape :", X_test_enc.shape)
print("Saved in data/processed/")
