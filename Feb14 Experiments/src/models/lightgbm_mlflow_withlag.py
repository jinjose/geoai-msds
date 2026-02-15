import pandas as pd
import numpy as np
import mlflow
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run_lightgbm_mlflow(feature_df, cutoff_name):

    years = sorted(feature_df["year"].unique())
    results = []

    for i in range(1, len(years)):

        train_years = years[:i]
        test_year = years[i]

        train_df = feature_df[feature_df["year"].isin(train_years)].copy()
        test_df = feature_df[feature_df["year"] == test_year].copy()

        # Lag-1 yield per county
        lag_yield = (
            train_df.sort_values("year")
            .groupby("county")["yield_bu_acre"]
            .last()
        )

        train_df["lag1_yield"] = train_df["county"].map(lag_yield)
        test_df["lag1_yield"] = test_df["county"].map(lag_yield)

        train_df = train_df.dropna()
        test_df = test_df.dropna()

        if len(test_df) < 5:
            continue

        X_train = train_df.drop(columns=["county", "year", "yield_bu_acre"])
        y_train = train_df["yield_bu_acre"]

        X_test = test_df.drop(columns=["county", "year", "yield_bu_acre"])
        y_test = test_df["yield_bu_acre"]

        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            random_state=42,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(
            np.abs((y_test - y_pred) / np.clip(y_test, 1e-6, None))
        ) * 100

        results.append({
            "year": test_year,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
        })

    results_df = pd.DataFrame(results)

    # Log to ACTIVE MLflow run
    if not results_df.empty:
        mlflow.log_param("model_lgbm", "lightgbm")
        mlflow.log_metric("lgbm_mae_mean", results_df["mae"].mean())
        mlflow.log_metric("lgbm_rmse_mean", results_df["rmse"].mean())
        mlflow.log_metric("lgbm_mape_mean", results_df["mape"].mean())
        mlflow.log_metric("lgbm_r2_mean", results_df["r2"].mean())

    return results_df
