import pandas as pd
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
from pathlib import Path
import numpy as np
import json

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import FORECAST_CUTOFFS
from models.model_registry import MODEL_CONFIGS
from analysis.plots import log_county_yield_forecast
from analysis.log_model_comparison_plots import log_comparison_plots


# ============================================================
# PROJECT ROOT
# ============================================================

def find_project_root(start: Path) -> Path:
    for p in [start.resolve()] + list(start.resolve().parents):
        if (p / "src").exists():
            return p
    raise RuntimeError("Project root not found")


PROJECT_ROOT = find_project_root(Path(__file__))
FEATURE_DIR = PROJECT_ROOT / "training-dataset" / "features_frozen"
EXPORT_DIR = PROJECT_ROOT / "exported_models"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

CUTOFF_STAGE_MAP = {
    "jun01": "Early Season Forecast",
    "jul01": "Mid Season Forecast",
    "aug01": "Late Season Forecast",
}

TARGET_COUNTIES = ["Boone", "Benton", "Marshall", "plymouth", "wayne", "appanoose"]

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
mlflow.set_experiment("iowa_corn_yield_forecasting_feb28")


def _is_lightgbm_model(model) -> bool:
    try:
        from lightgbm.sklearn import LGBMModel
        return isinstance(model, LGBMModel)
    except Exception:
        return model.__class__.__module__.startswith("lightgbm")


def _sanitize_name(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

for cutoff_key in FORECAST_CUTOFFS.keys():

    if cutoff_key not in CUTOFF_STAGE_MAP:
        continue

    forecast_stage = CUTOFF_STAGE_MAP[cutoff_key]
    feature_path = FEATURE_DIR / f"features_{cutoff_key}.csv"

    if not feature_path.exists():
        continue

    feature_df = pd.read_csv(feature_path).sort_values(["year", "county"])

    with mlflow.start_run(run_name=forecast_stage):

        mlflow.log_param("cutoff_key", cutoff_key)

        results_dict = {}

        # ====================================================
        # MODEL LOOP
        # ====================================================
        for model_cfg in MODEL_CONFIGS:

            if not model_cfg.get("enabled", False):
                continue

            model_name = model_cfg["name"]

            with mlflow.start_run(run_name=model_name, nested=True):

                result = model_cfg["func"](feature_df, cutoff_key)

                pred_df = None
                final_model = None

                if isinstance(result, tuple):
                    if len(result) >= 1:
                        pred_df = result[0]
                    if len(result) >= 2:
                        final_model = result[1]
                else:
                    pred_df = result

                if pred_df is None or pred_df.empty:
                    continue

                # ------------------------------------------------
                # Overall Performance Metrics
                # ------------------------------------------------
                overall_r2 = float(r2_score(pred_df["y_true"], pred_df["y_pred"]))
                overall_rmse = float(
                    np.sqrt(mean_squared_error(pred_df["y_true"], pred_df["y_pred"]))
                )
                overall_mae = float(
                    mean_absolute_error(pred_df["y_true"], pred_df["y_pred"])
                )
                overall_mape = float(
                    np.mean(
                        np.abs(
                            (pred_df["y_true"] - pred_df["y_pred"])
                            / np.clip(pred_df["y_true"], 1e-6, None)
                        )
                    ) * 100
                )

                mlflow.log_metric("val_r2", overall_r2)
                mlflow.log_metric("val_rmse", overall_rmse)
                mlflow.log_metric("val_mae", overall_mae)
                mlflow.log_metric("val_mape", overall_mape)

                # County-level forecast plots
                for county in TARGET_COUNTIES:
                    log_county_yield_forecast(pred_df, county, model_name)

                results_dict[model_name] = {
                    "pred_df": pred_df,
                    "rmse": overall_rmse,
                    "r2": overall_r2,
                    "mae": overall_mae,
                    "mape": overall_mape,
                    "final_model": final_model,
                }

        # ====================================================
        # SUMMARY TABLE
        # ====================================================
        summary_rows = []
        for model_name, metrics in results_dict.items():
            summary_rows.append(
                {
                    "model": model_name,
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "mape": metrics["mape"],
                }
            )

        summary_df = pd.DataFrame(summary_rows)

        summary_path = f"model_summary_{cutoff_key}.csv"
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(summary_path)

        # ====================================================
        # SELECT BEST MODEL
        # ====================================================
        best_model_name = None

        if not summary_df.empty:

            best_row = summary_df.loc[summary_df["rmse"].idxmin()]
            best_model_name = str(best_row["model"])

            mlflow.log_metric("best_rmse", float(best_row["rmse"]))
            mlflow.log_metric("best_r2", float(best_row["r2"]))
            mlflow.log_metric("best_mae", float(best_row["mae"]))
            mlflow.log_metric("best_mape", float(best_row["mape"]))
            mlflow.log_param("best_model", best_model_name)

        # ====================================================
        # EXPORT BEST MODEL
        # ====================================================
        if best_model_name and best_model_name in results_dict:

            best_final_model = results_dict[best_model_name].get("final_model")

            if best_final_model is not None:

                cutoff_export_dir = EXPORT_DIR / _sanitize_name(cutoff_key)
                cutoff_export_dir.mkdir(parents=True, exist_ok=True)

                export_path = cutoff_export_dir / _sanitize_name(best_model_name)

                if _is_lightgbm_model(best_final_model):
                    mlflow.lightgbm.save_model(best_final_model, path=str(export_path))
                else:
                    mlflow.sklearn.save_model(best_final_model, path=str(export_path))

                print(f"[OK] Exported best model for {cutoff_key} to: {export_path}")

                if hasattr(best_final_model, "feature_name_"):
                    expected_features = list(best_final_model.feature_name_)
                elif hasattr(best_final_model, "feature_names_in_"):
                    expected_features = list(best_final_model.feature_names_in_)
                else:
                    raise ValueError(
                        f"Model {best_model_name} does not expose feature names."
                    )

                feature_schema = {
                    "cutoff": cutoff_key,
                    "model_name": best_model_name,
                    "expected_features": expected_features,
                    "categorical_features": ["county"] if "county" in expected_features else [],
                    "target": "yield_bu_acre"
                }

                schema_path = export_path / "feature_schema.json"

                with open(schema_path, "w") as f:
                    json.dump(feature_schema, f, indent=4)

                print(f"[OK] Saved feature schema to: {schema_path}")

        # ====================================================
        # COMPARISON PLOTS
        # ====================================================
        plot_input = {
            model_name: metrics["pred_df"]
            for model_name, metrics in results_dict.items()
            if metrics["pred_df"] is not None and not metrics["pred_df"].empty
        }

        log_comparison_plots(results=plot_input, forecast_stage=forecast_stage)

print("\nTraining complete. Metrics logged and models exported.")
print(f"Export folder: {EXPORT_DIR}")