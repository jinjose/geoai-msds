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


def run_lightgbm_no_lag_mlflow(
    feature_df: pd.DataFrame,
    cutoff_name: str,
    target_col: str = "yield_bu_acre",
):

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

        drop_cols = [target_col, "county", "year", "lag1_yield"]
        feature_cols = [c for c in train_df.columns if c not in drop_cols]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col].values

        model = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        mape = float(np.mean(np.abs((y_test - preds) / np.clip(y_test, 1e-6, None))) * 100)

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
    # LOG VALIDATION METRICS
    # ======================================================
    if not yearly_df.empty:
        mlflow.log_metric("val_rmse_mean", float(yearly_df["rmse"].mean()))
        mlflow.log_metric("val_mape_mean", float(yearly_df["mape"].mean()))
        mlflow.log_metric("val_r2_mean", float(yearly_df["r2"].mean()))

    # ======================================================
    # TRAIN FINAL MODEL ON FULL DATA
    # ======================================================
    drop_cols = [target_col, "county", "year", "lag1_yield"]
    feature_cols = [c for c in feature_df.columns if c not in drop_cols]

    X_full = feature_df[feature_cols]
    y_full = feature_df[target_col]

    final_model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(X_full, y_full)

    # ======================================================
    # LOG MODEL
    # ======================================================
    signature = infer_signature(X_full, final_model.predict(X_full))
    input_example = X_full.head(5)

    mlflow.lightgbm.log_model(
        final_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
    )

    mlflow.log_dict({"features": feature_cols}, "feature_schema.json")
    mlflow.log_param("final_training_rows", int(len(feature_df)))
    mlflow.log_param("num_features", int(len(feature_cols)))

    mlflow.set_tag("model_name", "LightGBM-No-Lag")
    mlflow.set_tag("cutoff", cutoff_name)
    mlflow.set_tag("deployment_ready", "true")

    # ======================================================
    # SHAP ANALYSIS
    # ======================================================

    print("Running SHAP analysis...")

    shap_values = run_shap_analysis(
        model=final_model,
        X=X_full,
        model_name="LightGBM-No-Lag",
        cutoff_name=cutoff_name,
        return_values=True,
    )

    log_shap_year_contrast(
        shap_values=shap_values,
        X=X_full,
        years=feature_df["year"],
        cutoff_name=cutoff_name,
        model_name="LightGBM-No-Lag",
    )

    return yearly_df, prediction_df, final_model
