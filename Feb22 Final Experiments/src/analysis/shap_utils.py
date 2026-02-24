import shap
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import os


def run_shap_analysis(
    model,
    X,
    model_name: str,
    cutoff_name: str,
    max_display: int = 20,
    return_values: bool = False,
):
    """
    Runs global SHAP analysis for tree-based models.

    Logs:
    - SHAP summary plot
    - Mean absolute SHAP values (as CSV)
    - Top feature importance metrics
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Convert to absolute importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = (
        X.columns.to_series()
        .to_frame(name="feature")
        .assign(mean_abs_shap=mean_abs_shap)
        .sort_values("mean_abs_shap", ascending=False)
    )

    # -----------------------
    # Log SHAP importance CSV
    # -----------------------
    csv_path = f"shap_global_importance_{model_name}_{cutoff_name}.csv"
    importance_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path="shap")
    os.remove(csv_path)

    # -----------------------
    # Log SHAP summary plot
    # -----------------------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        show=False,
        max_display=max_display,
    )

    plot_path = f"shap_summary_{model_name}_{cutoff_name}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    mlflow.log_artifact(plot_path, artifact_path="shap")
    plt.close()
    os.remove(plot_path)

    # Log top feature as metric
    top_feature = importance_df.iloc[0]["feature"]
    mlflow.set_tag("top_shap_feature", top_feature)

    if return_values:
        return shap_values
