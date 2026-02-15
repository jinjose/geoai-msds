import pandas as pd
import mlflow
import mlflow.data
import mlflow.lightgbm
from pathlib import Path
import numpy as np

# ============================================================
# ROBUST PROJECT ROOT
# ============================================================
def find_project_root(start: Path) -> Path:
    for p in [start.resolve()] + list(start.resolve().parents):
        if (p / "src").exists():
            return p
    raise RuntimeError("Project root not found (expected 'src/' directory)")

PROJECT_ROOT = find_project_root(Path(__file__))
FEATURE_DIR = PROJECT_ROOT / "data" / "features_frozen"

# ============================================================
# IMPORTS
# ============================================================
from config import FORECAST_CUTOFFS, TARGET_COL
from models.model_registry import MODEL_CONFIGS

from analysis.plots import log_boone_actual_vs_pred
from analysis.log_model_comparison_plots import log_comparison_plots

# ============================================================
# FORECAST STAGE MAP
# ============================================================
CUTOFF_STAGE_MAP = {
    "jun01": "Early Season Forecast",
    "jul01": "Early Mid Season Forecast",
    "jul15": "Mid Season Forecast",
    "aug01": "Late Season Forecast",
}

# ============================================================
# MLFLOW SETUP
# ============================================================
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment("iowa_corn_yield_forecastingfeb14")

print(f"Tracking to: {MLRUNS_DIR}")

# ============================================================
# SAFE MEAN HELPER
# ============================================================
def safe_mean(df, col):
    return float(df[col].mean()) if df is not None and col in df.columns and not df.empty else np.nan

# ============================================================
# TRAINING LOOP
# ============================================================
for cutoff_key in FORECAST_CUTOFFS.keys():

    print(f"\n>>> Processing Cutoff: {cutoff_key} <<<")

    if cutoff_key not in CUTOFF_STAGE_MAP:
        raise ValueError(f"Cutoff '{cutoff_key}' not mapped in CUTOFF_STAGE_MAP.")

    forecast_stage = CUTOFF_STAGE_MAP[cutoff_key]

    feature_path = FEATURE_DIR / f"features_{cutoff_key}.csv"
    if not feature_path.exists():
        print(f"Skipping {cutoff_key}: File not found")
        continue

    feature_df = pd.read_csv(feature_path).sort_values(["year", "county"])

    # ========================================================
    # START PARENT RUN (FORECAST STAGE)
    # ========================================================
    with mlflow.start_run(run_name=forecast_stage):

        # ----------------------------------------------------
        # RUN METADATA
        # ----------------------------------------------------
        mlflow.log_param("cutoff_key", cutoff_key)
        mlflow.log_param("forecast_stage", forecast_stage)
        mlflow.log_param("target_column", TARGET_COL)
        mlflow.log_param("forecast_target_definition", "Final Annual Yield")
        mlflow.set_tag("forecast_target", "final_annual_yield")

        mlflow.log_param("num_counties", int(feature_df["county"].nunique()))
        mlflow.log_param(
            "year_range",
            f"{int(feature_df['year'].min())}-{int(feature_df['year'].max())}",
        )
        mlflow.set_tag("data_version", "features_frozen_v1")

        # ----------------------------------------------------
        # DATASET LOGGING
        # ----------------------------------------------------
        dataset = mlflow.data.from_pandas(
            feature_df,
            source=str(feature_path),
            name=f"{forecast_stage} Dataset",
        )
        mlflow.log_input(dataset)

        # ====================================================
        # MODEL EXECUTION (NESTED RUNS)
        # ====================================================
        results_dict = {}

        for model_cfg in MODEL_CONFIGS:

            if not model_cfg.get("enabled", False):
                continue

            model_name = model_cfg["name"]

            with mlflow.start_run(run_name=model_name, nested=True):

                mlflow.log_param("model_architecture", model_name)

                result = model_cfg["func"](feature_df, cutoff_key)

                final_model = None
                pred_df = None

                # Support: (metrics_df), (metrics_df, pred_df), (metrics_df, pred_df, final_model)
                if isinstance(result, tuple):
                    model_df = result[0]
                    if len(result) >= 2:
                        pred_df = result[1]
                    if len(result) >= 3:
                        final_model = result[2]
                else:
                    model_df = result

                # Always store results for comparison plots + summary
                results_dict[model_name] = model_df

                # Boone plot (if pred_df returned)
                if pred_df is not None and not pred_df.empty:
                    log_boone_actual_vs_pred(
                        pred_df,
                        cutoff_key,
                        model_name,
                    )

                    # Log the schema that model expects (if provided)
                    if "feature_cols" in model_cfg:
                        mlflow.log_dict(
                            {"features": model_cfg["feature_cols"]},
                            "feature_schema.json",
                        )

                    # Minimal training metadata
                    mlflow.log_param("final_training_rows", int(len(feature_df)))

        # ====================================================
        # SUMMARY TABLE
        # ====================================================
        summary_rows = []

        for model_cfg in MODEL_CONFIGS:
            if not model_cfg.get("enabled", False):
                continue

            model_name = model_cfg["name"]
            df = results_dict.get(model_name)
            if df is None or df.empty:
                continue

            metric_cols = model_cfg["metric_cols"]

            summary_rows.append(
                {
                    "model": model_name,
                    "mae_mean": safe_mean(df, metric_cols[0]),
                    "rmse_mean": safe_mean(df, metric_cols[1]),
                    "mape_mean": safe_mean(df, metric_cols[2]),
                    "r2_mean": safe_mean(df, metric_cols[3]),
                }
            )

        summary_df = pd.DataFrame(summary_rows)

        summary_path = f"model_summary_{cutoff_key}.csv"
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path)

        # ====================================================
        # LOG BEST MODEL (PARENT RUN)
        # ====================================================
        if not summary_df.empty:
            best_row = summary_df.loc[summary_df["rmse_mean"].idxmin()]

            mlflow.log_metric("best_rmse", float(best_row["rmse_mean"]))
            mlflow.log_metric("best_mape", float(best_row["mape_mean"]))
            mlflow.log_metric("best_r2", float(best_row["r2_mean"]))
            mlflow.log_param("best_model_architecture", str(best_row["model"]))

        # ====================================================
        # COMPARISON PLOTS (PARENT RUN)
        # ====================================================
        print("Generating comparison plots...")
        log_comparison_plots(
            results=results_dict,
            forecast_stage=forecast_stage,
        )

print("\nAll training runs complete. Run `mlflow ui` to view results.")
