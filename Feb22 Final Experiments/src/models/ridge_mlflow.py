import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


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


def run_ridge(feature_df, cutoff_key=None, alpha=1.0):
    years = sorted(feature_df["year"].unique())
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

        if train_df.empty or len(test_df) < 5:
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

        # Align columns
        X_train, X_test = X_train.align(
            X_test,
            join="left",
            axis=1,
            fill_value=0
        )

        y_train = train_df["yield_bu_acre"]
        y_test = test_df["yield_bu_acre"]

        # -----------------------------------------------------
        # Scale numeric columns
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
        # Store predictions ONLY
        # -----------------------------------------------------
        fold_pred_df = pd.DataFrame({
            "year": test_year,
            "county": test_df["county"].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
        })

        all_predictions.append(fold_pred_df)

    # -----------------------------------------------------
    # Final prediction dataframe
    # -----------------------------------------------------
    pred_df = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame(columns=["year", "county", "y_true", "y_pred"])
    )

    return pred_df