import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# FROZEN FEATURE SET FOR RIDGE
# =====================================================

RIDGE_FEATURES = [
    "rolling_3yr_mean",
    "ndvi_peak",
    "ndvi_slope",
    "temp_anomaly",
    "net_moisture_stress",
    "heat_days_gt32",
    "wind_severe_days_58_cutoff"
]


def run_ridge_mlflow(feature_df, cutoff_name, alpha=1.0):

    years = sorted(feature_df["year"].unique())
    yearly_results = []
    all_predictions = []

    for i in range(1, len(years)):

        train_years = years[:i]
        test_year = years[i]

        train_df = feature_df[
            feature_df["year"].isin(train_years)
        ].copy()

        test_df = feature_df[
            feature_df["year"] == test_year
        ].copy()

        # -----------------------------------------------------
        # Lag calculation (filtering consistency only)
        # -----------------------------------------------------
        lag_yield = (
            train_df.sort_values("year")
            .groupby("county")["yield_bu_acre"]
            .last()
        )

        train_df["lag1_yield"] = train_df["county"].map(lag_yield)
        test_df["lag1_yield"] = test_df["county"].map(lag_yield)

        train_df = train_df.dropna(subset=["lag1_yield"])
        test_df = test_df.dropna(subset=["lag1_yield"])

        if train_df.empty or len(test_df) < 5:
            print(f"Skipping year {test_year}: insufficient samples.")
            continue

        # -----------------------------------------------------
        # Build features INCLUDING county fixed effects
        # -----------------------------------------------------
        numeric_cols = RIDGE_FEATURES
        categorical_cols = ["county"]

        X_train = train_df[numeric_cols + categorical_cols].copy()
        X_test = test_df[numeric_cols + categorical_cols].copy()

        # One-hot encode county
        X_train = pd.get_dummies(
            X_train,
            columns=categorical_cols,
            drop_first=True
        )

        X_test = pd.get_dummies(
            X_test,
            columns=categorical_cols,
            drop_first=True
        )

        # Align columns (important for rolling backtesting)
        X_train, X_test = X_train.align(
            X_test,
            join="left",
            axis=1,
            fill_value=0
        )

        y_train = train_df["yield_bu_acre"]
        y_test = test_df["yield_bu_acre"]

        # -----------------------------------------------------
        # Scale ONLY numeric columns
        # -----------------------------------------------------
        scaler = StandardScaler()

        X_train[numeric_cols] = scaler.fit_transform(
            X_train[numeric_cols]
        )

        X_test[numeric_cols] = scaler.transform(
            X_test[numeric_cols]
        )

        # -----------------------------------------------------
        # Fit Ridge
        # -----------------------------------------------------
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mape = np.mean(
            np.abs((y_test - y_pred) / np.clip(y_test, 1e-6, None))
        ) * 100

        yearly_results.append({
            "year": test_year,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
        })

        # Store predictions
        fold_pred_df = pd.DataFrame({
            "year": test_year,
            "county": test_df["county"].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
        })

        all_predictions.append(fold_pred_df)

    # -----------------------------------------------------
    # Final dataframes
    # -----------------------------------------------------
    yearly_df = pd.DataFrame(yearly_results)

    pred_df = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )

    # -----------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------
    if not yearly_df.empty:

        mlflow.log_param("model_type", "ridge_with_county")
        mlflow.log_param("ridge_alpha", alpha)
        mlflow.log_param("num_years_validated", len(yearly_df))
        mlflow.log_param("num_features", len(RIDGE_FEATURES) + 1)

        mlflow.log_metric(
            "ridge_mae_mean",
            float(yearly_df["mae"].mean())
        )

        mlflow.log_metric(
            "ridge_rmse_mean",
            float(yearly_df["rmse"].mean())
        )

        mlflow.log_metric(
            "ridge_mape_mean",
            float(yearly_df["mape"].mean())
        )

        mlflow.log_metric(
            "ridge_r2_mean",
            float(yearly_df["r2"].mean())
        )

    return yearly_df, pred_df