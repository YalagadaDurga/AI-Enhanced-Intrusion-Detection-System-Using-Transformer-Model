from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted")
    }
    return metrics

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=[
        "Normal", "DoS", "Probe", "U2R", "R2L"
    ]))
