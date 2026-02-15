from pathlib import Path
import matplotlib.pyplot as plt
import mlflow


def log_boone_actual_vs_pred(
    pred_df,
    cutoff_key,
    model_name,
):

    boone_df = pred_df[pred_df["county"].str.lower() == "boone"].copy()

    if boone_df.empty:
        return

    boone_df = boone_df.sort_values("year")

    # Save locally first
    filename = f"boone_forecast_{model_name}.png"
    save_path = Path(filename)

    plt.figure(figsize=(10, 6))

    plt.plot(
        boone_df["year"],
        boone_df["y_true"],
        marker="o",
        linewidth=2,
        label="Observed Yield",
    )

    plt.plot(
        boone_df["year"],
        boone_df["y_pred"],
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Predicted Yield",
    )

    plt.xticks(boone_df["year"], rotation=45)
    plt.xlabel("Year")
    plt.ylabel("Yield (bu/acre)")
    plt.title(f"Boone County Final Yield Forecast\n{model_name}")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=60)
    plt.close()

    # 🔹 Log under "plots/" inside this model run
    mlflow.log_artifact(str(save_path), artifact_path="plots")

    # Optional: remove local file after logging
    save_path.unlink()
