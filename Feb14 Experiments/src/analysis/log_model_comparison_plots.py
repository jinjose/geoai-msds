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

        if "year" in df.columns:
            if "fold" in df.columns:
                mean_val = df["mae"].mean()
                plt.axhline(
                    y=mean_val,
                    label=f"{model_name} (Spatial Mean)",
                    linestyle="-.",
                    alpha=0.7,
                )
            else:
                plt.plot(
                    df["year"],
                    df["mae"],
                    marker="o",
                    label=model_name,
                )

        if "mae_guarded" in df.columns:
            plt.plot(
                df["year"],
                df["mae_guarded"],
                marker="o",
                linestyle="--",
                label=f"{model_name} (Guarded)",
            )

    # Force all years visible
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

        if "year" in df.columns:
            if "fold" in df.columns:
                mean_val = df["rmse"].mean()
                plt.axhline(
                    y=mean_val,
                    label=f"{model_name} (Spatial Mean)",
                    linestyle="-.",
                    alpha=0.7,
                )
            else:
                plt.plot(
                    df["year"],
                    df["rmse"],
                    marker="o",
                    label=model_name,
                )

        if "rmse_guarded" in df.columns:
            plt.plot(
                df["year"],
                df["rmse_guarded"],
                marker="o",
                linestyle="--",
                label=f"{model_name} (Guarded)",
            )

    # Force all years visible
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
