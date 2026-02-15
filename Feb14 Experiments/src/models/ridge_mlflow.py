import pandas as pd
import mlflow
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run_ridge_mlflow(feature_df, cutoff_name, alpha=1.0):
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

        # Safety: skip very small test sets
        if len(test_df) < 5:
            continue

        X_train = train_df.drop(columns=["county", "year", "yield_bu_acre"])
        y_train = train_df["yield_bu_acre"]

        X_test = test_df.drop(columns=["county", "year", "yield_bu_acre"])
        y_test = test_df["yield_bu_acre"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        # ---- VERSION-SAFE RMSE ----
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        results.append({
            "year": test_year,
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": rmse,
        })

    results_df = pd.DataFrame(results)

    # Log to ACTIVE MLflow run (nested handled by train.py)
    mlflow.log_param("model_ridge", "ridge")
    mlflow.log_param("ridge_alpha", alpha)

    mlflow.log_metric("ridge_mae_mean", results_df["mae"].mean())
    mlflow.log_metric("ridge_rmse_mean", results_df["rmse"].mean())

    return results_df
