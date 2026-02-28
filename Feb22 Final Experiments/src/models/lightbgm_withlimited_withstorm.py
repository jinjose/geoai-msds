import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

from analysis.shap_utils import run_shap_analysis
from analysis.shap_year_contrast import log_shap_year_contrast


# ============================================================
# MAIN MODEL
# ============================================================

def run_lightgbm_limited_features_storm(
        feature_df: pd.DataFrame,
        cutoff_name: str,
        target_col: str = "yield_bu_acre",
):

    df = feature_df.copy()
    df["county"] = df["county"].astype("category")
    years = sorted(df["year"].unique())

    feature_cols = [
        "county", "rolling_3yr_mean", "ndvi_peak", "ndvi_slope",
        "temp_anomaly", "net_moisture_stress",
        "heat_days_gt32", "wind_severe_days_58_cutoff"
    ]

    prediction_rows = []

    # =====================================================
    # WALK FORWARD
    # =====================================================
    for i in range(1, len(years)):

        train_years = years[:i]
        test_year = years[i]

        train_df = df[df["year"].isin(train_years)]
        test_df = df[df["year"] == test_year]

        if test_df.empty:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
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
            eval_set=[(X_test, y_test)],
            eval_metric="mae",
            categorical_feature=["county"],
            callbacks=[
                lgb.early_stopping(50, first_metric_only=True),
                lgb.log_evaluation(0)
            ]
        )

        preds = model.predict(X_test)

        fold_pred_df = pd.DataFrame({
            "year": test_year,
            "county": test_df["county"].values,
            "y_true": y_test,
            "y_pred": preds,
        })

        prediction_rows.append(fold_pred_df)

    prediction_df = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows else
        pd.DataFrame(columns=["year", "county", "y_true", "y_pred"])
    )

    # =====================================================
    # Compute metrics locally (FOR PLOTS ONLY)
    # =====================================================
    overall_mae = overall_rmse = overall_r2 = overall_mape = None

    if not prediction_df.empty:

        overall_r2 = float(r2_score(
            prediction_df["y_true"], prediction_df["y_pred"]
        ))

        overall_rmse = float(np.sqrt(mean_squared_error(
            prediction_df["y_true"], prediction_df["y_pred"]
        )))

        overall_mae = float(mean_absolute_error(
            prediction_df["y_true"], prediction_df["y_pred"]
        ))

        overall_mape = float(
            np.mean(
                np.abs(
                    (prediction_df["y_true"] - prediction_df["y_pred"]) /
                    np.clip(prediction_df["y_true"], 1e-6, None)
                )
            ) * 100
        )

    # =====================================================
    # FINAL MODEL (RETRAIN)
    # =====================================================
    X_full = df[feature_cols]
    y_full = df[target_col]

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

    final_model.fit(X_full, y_full, categorical_feature=["county"])

    signature = infer_signature(X_full, final_model.predict(X_full))
    mlflow.lightgbm.log_model(final_model, "model", signature=signature)

    # =====================================================
    # SHAP
    # =====================================================
    shap_values = run_shap_analysis(
        model=final_model,
        X=X_full,
        model_name="LightGBM-Limited-Storm",
        cutoff_name=cutoff_name,
        return_values=True
    )

    log_shap_year_contrast(
        shap_values=shap_values,
        X=X_full,
        years=df["year"],
        cutoff_name=cutoff_name,
        model_name="LightGBM-Limited-Storm"
    )

    # =====================================================
    # DIAGNOSTIC PLOTS RESTORED
    # =====================================================
    if not prediction_df.empty:

        log_production_summary_figure(
            prediction_df,
            shap_values,
            feature_cols,
            overall_mae,
            overall_mape,
            overall_r2,
            "production_summary_storm.png"
        )

        log_residual_diagnostics(
            prediction_df,
            "residual_diagnostics_storm.png"
        )

    return prediction_df, final_model

# ============================================================
# SUMMARY FIGURE
# ============================================================

def log_production_summary_figure(prediction_df, shap_values,
                                  feature_cols,
                                  overall_mae,
                                  overall_mape,
                                  overall_r2,
                                  artifact_name):

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    yearly = (
        prediction_df.groupby("year")[["y_true", "y_pred"]]
        .mean()
        .reset_index()
    )

    axes[0].plot(yearly["year"], yearly["y_true"], marker="o", label="Actual")
    axes[0].plot(yearly["year"], yearly["y_pred"], marker="o", label="Predicted")

    if overall_mae is not None:
        axes[0].set_title(
            f"Actual vs Predicted Yield\n"
            f"MAE: {overall_mae:.2f} bu | "
            f"MAPE: {overall_mape:.2f}% | "
            f"R²: {overall_r2:.2f}"
        )
    else:
        axes[0].set_title("Actual vs Predicted Yield")

    axes[0].legend()
    axes[0].grid(True)

    residuals = prediction_df["y_true"] - prediction_df["y_pred"]
    sns.histplot(residuals, bins=25, kde=True, ax=axes[1])
    axes[1].axvline(0, linestyle="--")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True)

    shap_abs_mean = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": shap_abs_mean
    }).sort_values("importance")

    axes[2].barh(shap_df["feature"], shap_df["importance"])
    axes[2].set_title("SHAP Feature Importance")

    plt.tight_layout()
    mlflow.log_figure(fig, artifact_name)
    plt.close(fig)


# ============================================================
# RESIDUAL DIAGNOSTICS
# ============================================================

def log_residual_diagnostics(prediction_df, artifact_name):

    if prediction_df.empty:
        return

    df = prediction_df.copy()
    df["residual"] = df["y_true"] - df["y_pred"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.histplot(df["residual"], bins=25, kde=True, ax=axes[0])
    axes[0].axvline(0, linestyle="--")
    axes[0].set_title("Overall Residual Distribution")
    axes[0].grid(True)

    county_order = (
        df.groupby("county")["residual"]
        .mean()
        .sort_values()
        .index
    )

    sns.boxplot(
        data=df,
        x="residual",
        y="county",
        order=county_order,
        ax=axes[1]
    )

    axes[1].axvline(0, linestyle="--")
    axes[1].set_title("Residual Distribution by County")

    pivot = df.pivot_table(
        index="county",
        columns="year",
        values="residual"
    )

    sns.heatmap(
        pivot,
        center=0,
        cmap="coolwarm",
        ax=axes[2]
    )

    axes[2].set_title("County-Year Residual Heatmap")

    plt.tight_layout()
    mlflow.log_figure(fig, artifact_name)
    plt.close(fig)