import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def log_comparison_plots(results: dict, forecast_stage: str):
    """
    results: dict[str, prediction_df]
    prediction_df must contain:
    ['county', 'year', 'y_true', 'y_pred']
    """

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    safe_name = forecast_stage.lower().replace(" ", "_")

    # ======================================================
    # Compute metrics for all models
    # ======================================================
    comparison_rows = []

    for model_name, pred_df in results.items():

        if pred_df.empty:
            continue

        y_true = pred_df["y_true"].to_numpy()
        y_pred = pred_df["y_pred"].to_numpy()

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = float(
            np.mean(
                np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))
            ) * 100
        )

        comparison_rows.append({
            "model": model_name,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "n_samples": len(pred_df)
        })

        # Log metrics
        mlflow.log_metric(f"{model_name.lower()}_mae", mae)
        mlflow.log_metric(f"{model_name.lower()}_rmse", rmse)
        mlflow.log_metric(f"{model_name.lower()}_mape", mape)
        mlflow.log_metric(f"{model_name.lower()}_r2", r2)
        mlflow.log_metric(f"{model_name.lower()}_n", len(pred_df))

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    # Log comparison table
    mlflow.log_table(comparison_df, "model_comparison.json")

    print("\n=== Model Comparison ===")
    print(comparison_df)

    # ======================================================
    # Create Bar Plots
    # ======================================================
    metrics = ["mae", "rmse", "mape", "r2"]

    for metric in metrics:

        plt.figure(figsize=(8, 5))

        plt.bar(
            comparison_df["model"],
            comparison_df[metric]
        )

        plt.title(f"{metric.upper()} Comparison – {forecast_stage}")
        plt.ylabel(metric.upper())
        plt.grid(axis="y", alpha=0.3)

        plot_path = plots_dir / f"{metric}_comparison_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(str(plot_path), artifact_path="plots")