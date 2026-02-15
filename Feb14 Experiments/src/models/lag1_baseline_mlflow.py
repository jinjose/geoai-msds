import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_lag1_baseline_mlflow(feature_df, cutoff_name):
    years = sorted(feature_df["year"].unique())
    results = []

    for i in range(1, len(years)):
        train_years = years[:i]
        test_year = years[i]

        train_df = feature_df[feature_df["year"].isin(train_years)]
        test_df = feature_df[feature_df["year"] == test_year].copy()

        # Lag-1 yield per county
        lag_yield = (
            train_df.sort_values("year")
            .groupby("county")["yield_bu_acre"]
            .last()
        )

        test_df["pred"] = test_df["county"].map(lag_yield)
        test_df = test_df.dropna(subset=["pred", "yield_bu_acre"])

        # Safety: skip very small test sets
        if len(test_df) < 5:
            continue

        y_true = test_df["yield_bu_acre"]
        y_pred = test_df["pred"]

        # ---- Metrics ----
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5

        # MAPE (safe division)
        mape = (abs((y_true - y_pred) / y_true).replace([float("inf")], 0).mean()) * 100

        # R²
        r2 = r2_score(y_true, y_pred)

        results.append({
            "year": test_year,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2
        })

    results_df = pd.DataFrame(results)

    # Log to ACTIVE MLflow run
    mlflow.log_param("model_lag1", "baseline")

    mlflow.log_metric("lag1_mae_mean", results_df["mae"].mean())
    mlflow.log_metric("lag1_rmse_mean", results_df["rmse"].mean())
    mlflow.log_metric("lag1_mape_mean", results_df["mape"].mean())
    mlflow.log_metric("lag1_r2_mean", results_df["r2"].mean())

    return results_df
