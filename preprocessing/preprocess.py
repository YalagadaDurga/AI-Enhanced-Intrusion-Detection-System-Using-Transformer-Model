# preprocessing/preprocess.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "kdd_train.csv")
test_path  = os.path.join(DATA_DIR, "kdd_test.csv")

# ------------------------------------------------
# Load data
# ------------------------------------------------
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

target_col = train.columns[-1]  # NSL-KDD label column

# ------------------------------------------------
# Attack → 5 class mapping
# ------------------------------------------------
attack_map = {
    "normal": "Normal",

    "neptune": "DoS", "smurf": "DoS", "back": "DoS",
    "teardrop": "DoS", "pod": "DoS", "land": "DoS",

    "satan": "Probe", "ipsweep": "Probe",
    "portsweep": "Probe", "nmap": "Probe",

    "guess_passwd": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "ftp_write": "R2L",
    "imap": "R2L", "phf": "R2L", "multihop": "R2L",

    "buffer_overflow": "U2R", "loadmodule": "U2R",
    "rootkit": "U2R", "perl": "U2R"
}

train[target_col] = train[target_col].str.lower().map(attack_map)
test[target_col]  = test[target_col].str.lower().map(attack_map)

train.dropna(inplace=True)
test.dropna(inplace=True)

class_map = {"Normal":0, "DoS":1, "Probe":2, "R2L":3, "U2R":4}
train[target_col] = train[target_col].map(class_map)
test[target_col]  = test[target_col].map(class_map)

# ------------------------------------------------
# Feature split
# ------------------------------------------------
cat_cols = ["protocol_type", "service", "flag"]
num_cols = [c for c in train.columns if c not in cat_cols + [target_col]]

X_train = train.drop(columns=[target_col])
y_train = train[target_col]

X_test  = test.drop(columns=[target_col])
y_test  = test[target_col]

# ------------------------------------------------
# Encoding + Scaling
# ------------------------------------------------
encoder = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

X_train = encoder.fit_transform(X_train)
X_test  = encoder.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ------------------------------------------------
# SAVE encoder & scaler  ⭐ IMPORTANT
# ------------------------------------------------
joblib.dump(encoder, os.path.join(BASE_DIR, "encoder.pkl"))
joblib.dump(scaler,  os.path.join(BASE_DIR, "scaler.pkl"))

# ------------------------------------------------
# SMOTE (TRAIN ONLY)
# ------------------------------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ------------------------------------------------
# Save arrays
# ------------------------------------------------
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)

print("✅ TRAIN preprocessing complete")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("Features:", X_train.shape[1])
