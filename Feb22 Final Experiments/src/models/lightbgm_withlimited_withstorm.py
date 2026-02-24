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


def run_lightgbm_limited_features_storm(
        feature_df: pd.DataFrame,
        cutoff_name: str,
        target_col: str = "yield_bu_acre",
):

    df = feature_df.copy()

    required_cols = ["county", "year", target_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found")

    df["county"] = df["county"].astype("category")
    years = sorted(df["year"].unique())

    limited_feature_cols = [
        "county", "rolling_3yr_mean", "ndvi_peak", "ndvi_slope",
        "temp_anomaly", "net_moisture_stress", "heat_days_gt32",
        "wind_severe_days_58_cutoff"
    ]

    yearly_results = []
    prediction_rows = []

    # =====================================================
    # WALK-FORWARD VALIDATION
    # =====================================================
    for i in range(1, len(years)):

        train_years = years[:i]
        test_year = years[i]

        train_df = df[df["year"].isin(train_years)]
        test_df = df[df["year"] == test_year]

        if test_df.empty:
            continue

        X_train = train_df[limited_feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[limited_feature_cols]
        y_test = test_df[target_col].values

        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=40,
            reg_lambda=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=["train", "valid"],
            eval_metric="mae",
            categorical_feature=["county"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        preds = model.predict(X_test)

        # ---- Train vs Validation Gap ----
        evals = model.evals_result_
        best_iter = model.best_iteration_
        train_mae = evals["train"]["l1"][best_iter - 1]
        valid_mae = evals["valid"]["l1"][best_iter - 1]
        mae_gap = valid_mae - train_mae

        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        mape = float(
            np.mean(np.abs((y_test - preds) / np.clip(y_test, 1e-6, None))) * 100
        )

        yearly_results.append({
            "year": test_year,
            "mae": mae,
            "mae_train": train_mae,
            "mae_valid": valid_mae,
            "mae_gap": mae_gap,
            "rmse": rmse,
            "mape": mape,
            "r2": r2
        })

        res_row = X_test.copy()
        res_row["year"] = test_year
        res_row["y_true"] = y_test
        res_row["y_pred"] = preds
        prediction_rows.append(res_row)

    yearly_df = pd.DataFrame(yearly_results)
    prediction_df = pd.concat(prediction_rows, ignore_index=True)

    # =====================================================
    # OVERALL OUT-OF-SAMPLE METRICS
    # =====================================================
    if not prediction_df.empty:

        overall_mae = float(
            mean_absolute_error(prediction_df["y_true"], prediction_df["y_pred"])
        )
        overall_rmse = float(
            np.sqrt(mean_squared_error(prediction_df["y_true"], prediction_df["y_pred"]))
        )
        overall_r2 = float(
            r2_score(prediction_df["y_true"], prediction_df["y_pred"])
        )

        mlflow.log_metric("val_mae_overall", overall_mae)
        mlflow.log_metric("val_rmse_overall", overall_rmse)
        mlflow.log_metric("val_r2_overall", overall_r2)

    # =====================================================
    # FINAL MODELS (POINT + QUANTILE)
    # =====================================================
    X_full = df[limited_feature_cols]
    y_full = df[target_col]

    # Point forecast model
    final_model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        reg_lambda=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Lower quantile model (1%)
    final_model_low = LGBMRegressor(
        objective="quantile",
        alpha=0.01,
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        reg_lambda=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X_full, y_full, categorical_feature=["county"])
    final_model_low.fit(X_full, y_full, categorical_feature=["county"])

    # =====================================================
    # RISK DETECTION USING PREDICTION SPREAD
    # =====================================================
    if not prediction_df.empty:

        prediction_df["y_pred_low"] = final_model_low.predict(
            prediction_df[limited_feature_cols]
        )

        prediction_df["prediction_spread"] = (
            prediction_df["y_pred"] - prediction_df["y_pred_low"]
        )

        spread_threshold = prediction_df["prediction_spread"].quantile(0.75)

        prediction_df["risk_level"] = np.where(
            prediction_df["prediction_spread"] >= spread_threshold,
            "High Risk",
            "Normal"
        )

        high_risk_count = int(
            (prediction_df["risk_level"] == "High Risk").sum()
        )

        mlflow.log_metric("high_risk_count", high_risk_count)

    # =====================================================
    # MODEL LOGGING
    # =====================================================
    signature = infer_signature(X_full, final_model.predict(X_full))

    mlflow.lightgbm.log_model(
        final_model,
        artifact_path="model",
        signature=signature
    )

    shap_values = run_shap_analysis(
        model=final_model,
        X=X_full,
        model_name="LightGBM-Limited-Features-Storm",
        cutoff_name=cutoff_name,
        return_values=True
    )

    log_shap_year_contrast(
        shap_values=shap_values,
        X=X_full,
        years=df["year"],
        cutoff_name=cutoff_name,
        model_name="LightGBM-Limited-Features-Storm"
    )

    return yearly_df, prediction_df, final_model