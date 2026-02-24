import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path


def log_comparison_plots(results: dict, forecast_stage: str):

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    safe_name = forecast_stage.lower().replace(" ", "_")

    # --------------------------------------------------
    # Collect all unique years across models
    # --------------------------------------------------
    all_years = sorted(
        set().union(*[
            set(df["year"].unique())
            for df in results.values()
            if not df.empty and "year" in df.columns
        ])
    )

    # ==================================================
    # MAE PLOT
    # ==================================================
    plt.figure(figsize=(10, 6))
    for model_name, df in results.items():
        if df.empty:
            continue
        if "mae" in df.columns:
            plt.plot(df["year"], df["mae"], marker="o", label=model_name)
    if all_years:
        plt.xticks(all_years, rotation=45)
    plt.xlabel("Test Year")
    plt.ylabel("Mean Absolute Error (bu/acre)")
    plt.title(f"MAE Comparison – {forecast_stage}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    mae_path = plots_dir / f"mae_comparison_{safe_name}.png"
    plt.savefig(mae_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(mae_path), artifact_path="plots")

    # ==================================================
    # RMSE PLOT
    # ==================================================
    plt.figure(figsize=(10, 6))
    for model_name, df in results.items():
        if df.empty:
            continue
        if "rmse" in df.columns:
            plt.plot(df["year"], df["rmse"], marker="o", label=model_name)
    if all_years:
        plt.xticks(all_years, rotation=45)
    plt.xlabel("Test Year")
    plt.ylabel("Root Mean Squared Error (bu/acre)")
    plt.title(f"RMSE Comparison – {forecast_stage}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    rmse_path = plots_dir / f"rmse_comparison_{safe_name}.png"
    plt.savefig(rmse_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(rmse_path), artifact_path="plots")

    # ==================================================
    # MAPE PLOT
    # ==================================================
    plt.figure(figsize=(10, 6))
    for model_name, df in results.items():
        if df.empty:
            continue
        if "mape" in df.columns:
            plt.plot(df["year"], df["mape"], marker="o", label=model_name)
    if all_years:
        plt.xticks(all_years, rotation=45)
    plt.xlabel("Test Year")
    plt.ylabel("Mean Absolute Percentage Error (%)")
    plt.title(f"MAPE Comparison – {forecast_stage}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    mape_path = plots_dir / f"mape_comparison_{safe_name}.png"
    plt.savefig(mape_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(mape_path), artifact_path="plots")

    # ==================================================
    # R2 PLOT (NEW)
    # ==================================================
    plt.figure(figsize=(10, 6))
    for model_name, df in results.items():
        if df.empty:
            continue
        if "r2" in df.columns:
            plt.plot(df["year"], df["r2"], marker="o", label=model_name)
    if all_years:
        plt.xticks(all_years, rotation=45)
    plt.xlabel("Test Year")
    plt.ylabel("R² Score")
    plt.title(f"R² Comparison – {forecast_stage}")
    plt.axhline(y=0, linestyle="--", alpha=0.5)  # zero reference line
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    r2_path = plots_dir / f"r2_comparison_{safe_name}.png"
    plt.savefig(r2_path, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(r2_path), artifact_path="plots")