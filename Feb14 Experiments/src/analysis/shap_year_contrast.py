import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import os


def log_shap_year_contrast(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    years: pd.Series,
    cutoff_name: str,
    model_name: str,
    year_a: int = 2023,
    year_b: int = 2024,
    stress_features=None,
    top_n: int = 10,
):
    """
    Compares SHAP importance between two years.
    Logs:
    - CSV comparison
    - Bar plot
    - Stress dominance ratios

    Safe for:
    - Full feature models
    - Reduced feature models
    - Future architectures
    """

    # -----------------------------------------------------
    # Default Stress Features
    # -----------------------------------------------------
    if stress_features is None:
        stress_features = [
            "temp_anomaly",
            "heat_days_gt32",
            "heat_rain_stress_idx",
            "ndvi_drop_rate",
            "net_moisture_stress",
        ]

    # -----------------------------------------------------
    # Create SHAP dataframe
    # -----------------------------------------------------
    shap_df = pd.DataFrame(
        np.abs(shap_values),
        columns=X.columns,
        index=X.index,
    )

    shap_df["year"] = years.values

    # -----------------------------------------------------
    # Ensure both years exist
    # -----------------------------------------------------
    if year_a not in shap_df["year"].unique():
        print(f"Year {year_a} not found. Skipping SHAP contrast.")
        return None

    if year_b not in shap_df["year"].unique():
        print(f"Year {year_b} not found. Skipping SHAP contrast.")
        return None

    # -----------------------------------------------------
    # Compute mean SHAP per year
    # -----------------------------------------------------
    shap_a = (
        shap_df[shap_df["year"] == year_a]
        .drop(columns="year")
        .mean()
    )

    shap_b = (
        shap_df[shap_df["year"] == year_b]
        .drop(columns="year")
        .mean()
    )

    # -----------------------------------------------------
    # Comparison Table
    # -----------------------------------------------------
    comparison_df = pd.concat(
        [shap_a, shap_b],
        axis=1,
        keys=[str(year_a), str(year_b)],
    )

    comparison_df["ratio_b_to_a"] = (
        comparison_df[str(year_b)]
        / np.clip(comparison_df[str(year_a)], 1e-9, None)
    )

    comparison_df = comparison_df.sort_values(
        by=str(year_a), ascending=False
    )

    # -----------------------------------------------------
    # Safe Stress Dominance Ratio
    # -----------------------------------------------------
    # Keep only stress features that exist in this model
    stress_features_present = [
        f for f in stress_features if f in shap_a.index
    ]

    if len(stress_features_present) > 0 and shap_a.sum() > 0:
        stress_ratio_a = (
            shap_a[stress_features_present].sum()
            / shap_a.sum()
        )
    else:
        stress_ratio_a = 0.0

    if len(stress_features_present) > 0 and shap_b.sum() > 0:
        stress_ratio_b = (
            shap_b[stress_features_present].sum()
            / shap_b.sum()
        )
    else:
        stress_ratio_b = 0.0

    mlflow.log_metric(
        f"shap_stress_ratio_{year_a}",
        float(stress_ratio_a),
    )

    mlflow.log_metric(
        f"shap_stress_ratio_{year_b}",
        float(stress_ratio_b),
    )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------
    csv_path = (
        f"shap_year_contrast_{model_name}_{cutoff_name}_"
        f"{year_a}_vs_{year_b}.csv"
    )

    comparison_df.to_csv(csv_path)
    mlflow.log_artifact(csv_path, artifact_path="shap")
    os.remove(csv_path)

    # -----------------------------------------------------
    # Plot Top Features
    # -----------------------------------------------------
    top_features = comparison_df.head(top_n)

    plt.figure(figsize=(8, 5))
    top_features[[str(year_a), str(year_b)]].plot(
        kind="barh"
    )

    plt.title(
        f"SHAP Feature Dominance: {year_a} vs {year_b}"
    )
    plt.xlabel("Mean |SHAP value|")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = (
        f"shap_year_contrast_{model_name}_{cutoff_name}_"
        f"{year_a}_vs_{year_b}.png"
    )

    plt.savefig(plot_path, dpi=150)
    mlflow.log_artifact(plot_path, artifact_path="shap")
    plt.close()
    os.remove(plot_path)

    return comparison_df
