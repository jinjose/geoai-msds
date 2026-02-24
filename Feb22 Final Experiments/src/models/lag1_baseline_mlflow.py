import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_lag1_baseline_mlflow(feature_df, cutoff_name):

    years = sorted(feature_df["year"].unique())

    yearly_results = []
    all_predictions = []

    for i in range(1, len(years)):

        train_years = years[:i]
        test_year = years[i]

        train_df = feature_df[
            feature_df["year"].isin(train_years)
        ]

        test_df = feature_df[
            feature_df["year"] == test_year
        ].copy()

        # -------------------------------------------------
        # Lag-1 prediction per county
        # -------------------------------------------------
        lag_yield = (
            train_df.sort_values("year")
            .groupby("county")["yield_bu_acre"]
            .last()
        )

        test_df["pred"] = test_df["county"].map(lag_yield)

        test_df = test_df.dropna(
            subset=["pred", "yield_bu_acre"]
        )

        if len(test_df) < 5:
            continue

        y_true = test_df["yield_bu_acre"]
        y_pred = test_df["pred"]

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mape = (
            abs((y_true - y_pred) / y_true)
            .replace([float("inf")], 0)
            .mean()
        ) * 100

        r2 = r2_score(y_true, y_pred)

        yearly_results.append({
            "year": test_year,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2
        })

        # -------------------------------------------------
        # Store predictions for overall scoring
        # -------------------------------------------------
        fold_pred_df = pd.DataFrame({
            "year": test_year,
            "county": test_df["county"].values,
            "y_true": y_true.values,
            "y_pred": y_pred.values,
        })

        all_predictions.append(fold_pred_df)

    # -------------------------------------------------
    # Build final outputs
    # -------------------------------------------------
    yearly_df = pd.DataFrame(yearly_results)

    pred_df = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )

    # -------------------------------------------------
    # MLflow logging
    # -------------------------------------------------
    if not yearly_df.empty:

        mlflow.log_param("model_type", "lag1_baseline")

        mlflow.log_metric(
            "lag1_mae_mean",
            float(yearly_df["mae"].mean())
        )

        mlflow.log_metric(
            "lag1_rmse_mean",
            float(yearly_df["rmse"].mean())
        )

        mlflow.log_metric(
            "lag1_mape_mean",
            float(yearly_df["mape"].mean())
        )

        mlflow.log_metric(
            "lag1_r2_mean",
            float(yearly_df["r2"].mean())
        )

    return yearly_df, pred_df