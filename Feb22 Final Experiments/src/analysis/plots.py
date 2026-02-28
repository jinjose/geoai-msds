from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import numpy as np


def log_county_yield_forecast(pred_df, county_name, model_name):
    """Generates professional risk-detection plots focused on 2019-2024 with fixed axes."""

    county_df = pred_df[pred_df["county"].str.lower() == county_name.lower()].copy()
    display_df = county_df[county_df["year"] >= 2019].sort_values("year")

    if display_df.empty:
        return

    # ---------------------------------------------------
    # FIXED AXIS LOGIC
    # ---------------------------------------------------

    # Fixed X range
    min_year = 2019
    max_year = 2024

    # Global Y range across ALL counties for consistency
    y_values = []

    if "y_true" in pred_df.columns:
        y_values.extend(pred_df["y_true"].values)

    if "y_pred" in pred_df.columns:
        y_values.extend(pred_df["y_pred"].values)

    if "y_pred_low" in pred_df.columns:
        y_values.extend(pred_df["y_pred_low"].values)

    y_min = np.floor(min(y_values) - 5)
    y_max = np.ceil(max(y_values) + 5)

    # ---------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------

    save_path = Path(f"{county_name.lower()}_forecast_{model_name}.png")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)

    ax.plot(display_df["year"], display_df["y_true"],
            color='#003f5c', marker='o',
            markersize=8, linewidth=2.5,
            label="Observed Yield", zorder=5)

    ax.plot(display_df["year"], display_df["y_pred"],
            color='#ffa600', marker='s',
            linestyle='--', linewidth=2,
            label="Mean Prediction", zorder=4)

    if "y_pred_low" in display_df.columns:
        ax.fill_between(display_df["year"],
                        display_df["y_pred_low"],
                        display_df["y_pred"],
                        color='#ffa600', alpha=0.15,
                        label="Risk Uncertainty Area", zorder=2)

        ax.plot(display_df["year"], display_df["y_pred_low"],
                color='#d45087',
                linestyle=':',
                linewidth=1.5,
                label="Stress Case",
                alpha=0.8,
                zorder=3)

    if "risk_level" in display_df.columns:
        for _, row in display_df.iterrows():
            if row["risk_level"] == "High Risk":
                ax.axvspan(row["year"] - 0.3,
                           row["year"] + 0.3,
                           color='#d45087',
                           alpha=0.08,
                           zorder=1)

                ax.annotate(
                    'HIGH RISK',
                    xy=(row["year"], row["y_pred"]),
                    xytext=(0, 12),
                    textcoords='offset points',
                    color='#d45087',
                    fontweight='bold',
                    ha='center',
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='#d45087')
                )

    # ---------------------------------------------------
    # APPLY FIXED AXES
    # ---------------------------------------------------

    ax.set_xlim(min_year - 0.5, max_year + 0.5)
    ax.set_ylim(y_min, y_max)

    ax.set_title(
        f"Yield Prediction for: {county_name.title()} (Focus: 2019-2024)",
        fontsize=15,
        fontweight='bold',
        pad=15
    )

    ax.set_ylabel("Yield (bu/acre)", fontsize=11)
    ax.set_xticks(range(min_year, max_year + 1))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=False, loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    mlflow.log_artifact(str(save_path), artifact_path="plots")
    save_path.unlink()