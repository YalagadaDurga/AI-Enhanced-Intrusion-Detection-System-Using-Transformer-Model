import shap
import torch
import matplotlib.pyplot as plt
import numpy as np


def shap_explain(model, X_background, X_explain, feature_names):
    """
    Generates SHAP explanation plot for a single prediction
    """

    # Wrapper for model prediction
    def model_predict(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(x_tensor)
        return outputs.numpy()

    # Convert tensors to numpy
    background_np = X_background.numpy()
    explain_np = X_explain.numpy()

    # Create SHAP explainer
    explainer = shap.KernelExplainer(
        model_predict,
        background_np
    )

    # Compute SHAP values
    shap_values = explainer.shap_values(explain_np, nsamples=100)

    # Plot
    plt.figure(figsize=(10, 5))
    shap.summary_plot(
        shap_values,
        explain_np,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    return plt.gcf()
