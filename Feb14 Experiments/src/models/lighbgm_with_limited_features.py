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


def run_lightgbm_limited_features_mlflow(
    feature_df: pd.DataFrame,
    cutoff_name: str,
    target_col: str = "yield_bu_acre",
):

    # =====================================================
    # COPY DATA (DO NOT MUTATE SCHEMA)
    # =====================================================
    df = feature_df.copy()

    required_cols = ["county", "year", target_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in feature_df")

    # Lock categorical levels globally
    df["county"] = df["county"].astype("category")
    all_counties = df["county"].cat.categories

    years = sorted(df["year"].unique())
    yearly_results = []
    prediction_rows = []

    # =====================================================
    # LIMITED FEATURE SET (Statistically Reduced)
    # =====================================================
    limited_feature_cols = [
        "county",
        "rolling_3yr_mean",
        "ndvi_peak",
        "ndvi_slope",
        "temp_anomaly",
        "net_moisture_stress",
        "heat_days_gt32"
    ]

    missing = [c for c in limited_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing limited features: {missing}")

    # =====================================================
    # WALK-FORWARD VALIDATION
    # =====================================================
    for i in range(1, len(years)):

        train_years = years[:i]
        test_year = years[i]

        train_df = df[df["year"].isin(train_years)].copy()
        test_df = df[df["year"] == test_year].copy()

        if test_df.empty:
            continue

        X_train = train_df[limited_feature_cols].copy()
        y_train = train_df[target_col]

        X_test = test_df[limited_feature_cols].copy()
        y_test = test_df[target_col].values

        # Ensure categorical consistency
        X_train["county"] = pd.Categorical(
            X_train["county"], categories=all_counties
        )
        X_test["county"] = pd.Categorical(
            X_test["county"], categories=all_counties
        )

        model = LGBMRegressor(
            n_estimators=1500,
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
            categorical_feature=["county"],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
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

    # =====================================================
    # LOG VALIDATION METRICS
    # =====================================================
    if not yearly_df.empty:
        mlflow.log_metric("val_rmse_mean", float(yearly_df["rmse"].mean()))
        mlflow.log_metric("val_mape_mean", float(yearly_df["mape"].mean()))
        mlflow.log_metric("val_r2_mean", float(yearly_df["r2"].mean()))

    # =====================================================
    # TRAIN FINAL MODEL ON FULL DATA
    # =====================================================
    X_full = df[limited_feature_cols].copy()
    y_full = df[target_col]

    X_full["county"] = pd.Categorical(
        X_full["county"], categories=all_counties
    )

    final_model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(
        X_full,
        y_full,
        categorical_feature=["county"],
    )

    # =====================================================
    # LOG MODEL
    # =====================================================
    signature = infer_signature(X_full, final_model.predict(X_full))

    mlflow.lightgbm.log_model(
        final_model,
        artifact_path="model",
        signature=signature,
    )

    mlflow.log_dict({"features": limited_feature_cols}, "feature_schema.json")
    mlflow.log_param("final_training_rows", int(len(df)))
    mlflow.log_param("num_features", int(len(limited_feature_cols)))

    mlflow.set_tag("model_name", "LightGBM-Limited-Features")
    mlflow.set_tag("cutoff", cutoff_name)
    mlflow.set_tag("deployment_ready", "true")

    # =====================================================
    # SHAP ANALYSIS
    # =====================================================
    print("Running SHAP analysis (Limited Features)...")

    shap_values = run_shap_analysis(
        model=final_model,
        X=X_full,
        model_name="LightGBM-Limited-Features",
        cutoff_name=cutoff_name,
        return_values=True,
    )

    log_shap_year_contrast(
        shap_values=shap_values,
        X=X_full,
        years=df["year"],
        cutoff_name=cutoff_name,
        model_name="LightGBM-Limited-Features",
    )

    return yearly_df, prediction_df, final_model
