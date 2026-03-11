import torch
import numpy as np

def predict(model, X, device):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X)
        _, preds = torch.max(outputs, 1)

    return preds.cpu().numpy()

def decode_predictions(preds):
    mapping = {
        0: "Normal",
        1: "DoS",
        2: "Probe",
        3: "U2R",
        4: "R2L"
    }
    return [mapping[p] for p in preds]
