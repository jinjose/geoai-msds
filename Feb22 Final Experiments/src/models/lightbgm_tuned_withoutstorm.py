import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import lightgbm as lgb

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

from analysis.shap_utils import run_shap_analysis
from analysis.shap_year_contrast import log_shap_year_contrast


FEATURES_NO_LAG = [
    "county",
    "rolling_3yr_mean",
    "ndvi_peak",
    "ndvi_slope",
    "temp_anomaly",
    "net_moisture_stress",
    "heat_days_gt32",
]

TUNED_PARAMS_NO_STORM = dict(
    n_estimators=1500,
    learning_rate=0.02,
    num_leaves=15,
    max_depth=4,
    min_child_samples=25,
    min_child_weight=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=5.0,
    min_split_gain=0.05,
    random_state=42,
    n_jobs=-1,
)


def run_lightgbm_tuned_withoutstorm(
    feature_df: pd.DataFrame,
    cutoff_name: str,
    target_col: str = "yield_bu_acre",
):

    feature_df = feature_df.copy()

    if target_col not in feature_df.columns:
        raise ValueError(f"{target_col} not found in dataframe")

    missing_features = [c for c in FEATURES_NO_LAG if c not in feature_df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    feature_df["county"] = feature_df["county"].astype("category")
    all_counties = feature_df["county"].cat.categories

    years = sorted(feature_df["year"].unique())
    yearly_results = []
    prediction_rows = []

    # ======================================================
    # WALK-FORWARD VALIDATION
    # ======================================================
    for i in range(1, len(years)):

        train_years = years[:i]
        test_year = years[i]

        train_df = feature_df[feature_df["year"].isin(train_years)].copy()
        test_df = feature_df[feature_df["year"] == test_year].copy()

        if test_df.empty:
            continue

        X_train = train_df[FEATURES_NO_LAG].copy()
        y_train = train_df[target_col]

        X_test = test_df[FEATURES_NO_LAG].copy()
        y_test = test_df[target_col].values

        X_train["county"] = pd.Categorical(X_train["county"], categories=all_counties)
        X_test["county"] = pd.Categorical(X_test["county"], categories=all_counties)

        model = LGBMRegressor(**TUNED_PARAMS_NO_STORM)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=["train", "valid"],
            eval_metric="mae",
            categorical_feature=["county"],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0),
            ],
        )

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        mape = float(
            np.mean(np.abs((y_test - preds) / np.clip(y_test, 1e-6, None))) * 100
        )

        yearly_results.append(
            {"year": test_year, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}
        )

        prediction_rows.append(
            pd.DataFrame(
                {
                    "county": test_df["county"].values,
                    "year": test_year,
                    "y_true": y_test,
                    "y_pred": preds,
                }
            )
        )

    yearly_df = pd.DataFrame(yearly_results)

    prediction_df = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame(columns=["county", "year", "y_true", "y_pred"])
    )

    # ======================================================
    # OVERALL OUT-OF-SAMPLE METRICS (PRIMARY KPI)
    # ======================================================
    if not prediction_df.empty:

        overall_r2 = float(
            r2_score(
                prediction_df["y_true"],
                prediction_df["y_pred"]
            )
        )

        overall_rmse = float(
            np.sqrt(
                mean_squared_error(
                    prediction_df["y_true"],
                    prediction_df["y_pred"]
                )
            )
        )

        overall_mae = float(
            mean_absolute_error(
                prediction_df["y_true"],
                prediction_df["y_pred"]
            )
        )

        overall_mape = float(
            np.mean(
                np.abs(
                    (prediction_df["y_true"] - prediction_df["y_pred"])
                    / np.clip(prediction_df["y_true"], 1e-6, None)
                )
            ) * 100
        )

    else:
        overall_r2 = np.nan
        overall_rmse = np.nan
        overall_mae = np.nan
        overall_mape = np.nan

    # Log stable metrics
    mlflow.log_metric("val_r2_overall", overall_r2)
    mlflow.log_metric("val_rmse_overall", overall_rmse)
    mlflow.log_metric("val_mae_overall", overall_mae)
    mlflow.log_metric("val_mape_overall", overall_mape)

    # ======================================================
    # MEAN YEARLY METRICS (REFERENCE ONLY)
    # ======================================================
    if not yearly_df.empty:
        mlflow.log_metric("val_r2_mean", float(yearly_df["r2"].mean()))
        mlflow.log_metric("val_rmse_mean", float(yearly_df["rmse"].mean()))
        mlflow.log_metric("val_mape_mean", float(yearly_df["mape"].mean()))

    # ======================================================
    # TRAIN FINAL MODEL ON FULL DATA
    # ======================================================
    X_full = feature_df[FEATURES_NO_LAG].copy()
    y_full = feature_df[target_col]

    X_full["county"] = pd.Categorical(X_full["county"], categories=all_counties)

    final_model = LGBMRegressor(**TUNED_PARAMS_NO_STORM)

    final_model.fit(
        X_full,
        y_full,
        categorical_feature=["county"],
    )

    signature = infer_signature(X_full, final_model.predict(X_full))

    mlflow.lightgbm.log_model(
        final_model,
        artifact_path="model",
        signature=signature,
    )

    mlflow.set_tag("model_name", "LightGBM-Tuned-No-Storm")
    mlflow.set_tag("cutoff", cutoff_name)
    mlflow.set_tag("deployment_ready", "true")

    return yearly_df, prediction_df, final_model