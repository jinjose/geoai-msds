import pandas as pd


def run_lag1_baseline(feature_df, cutoff_name=None):

    years = sorted(feature_df["year"].unique())
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

        test_df["y_pred"] = test_df["county"].map(lag_yield)

        test_df = test_df.dropna(
            subset=["y_pred", "yield_bu_acre"]
        )

        if len(test_df) < 5:
            continue

        # -------------------------------------------------
        # Store predictions only
        # -------------------------------------------------
        fold_pred_df = pd.DataFrame({
            "year": test_year,
            "county": test_df["county"].values,
            "y_true": test_df["yield_bu_acre"].values,
            "y_pred": test_df["y_pred"].values,
        })

        all_predictions.append(fold_pred_df)

    # -------------------------------------------------
    # Final prediction dataframe
    # -------------------------------------------------
    pred_df = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame(columns=["year", "county", "y_true", "y_pred"])
    )

    return pred_df