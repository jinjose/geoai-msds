import numpy as np
import pandas as pd
import logging
from utils import cutoff_mask

logger = logging.getLogger(__name__)


def feature_table_logic(
    yield_df: pd.DataFrame,
    ndvi: pd.DataFrame,
    wx: pd.DataFrame,
    storm_df: pd.DataFrame,
    cutoff_month: int,
    cutoff_day: int,
    temp_hist_mean: pd.Series,
    mode: str = "historical",
) -> pd.DataFrame:
    """
    historical → training features (includes target + lag1)
    live       → inference features (prediction rules, no target)
    """

    yield_hist = yield_df.sort_values(["county", "year"]).copy()

    valid_years = sorted(set(ndvi["year"]).intersection(set(wx["year"])))
    if not valid_years:
        logger.warning("No overlapping NDVI/WX years found.")
        return pd.DataFrame()

    # ======================================================
    # HISTORICAL MODE
    # ======================================================
    if mode == "historical":

        yield_hist["lag1_yield"] = (
            yield_hist.groupby("county")["yield_bu_acre"].shift(1)
        )

        yield_hist["rolling_3yr_mean"] = (
            yield_hist.groupby("county")["yield_bu_acre"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )

        yield_hist = yield_hist[yield_hist["year"].isin(valid_years)]

    # ======================================================
    # LIVE MODE (Prediction Rules)
    # ======================================================
    else:
        prediction_year = max(valid_years)
        logger.info(f"Live mode → applying prediction rules for {prediction_year}")

        prediction_rows = []

        for county in yield_hist["county"].unique():

            hist = (
                yield_hist[yield_hist["county"] == county]
                .sort_values("year")
            )

            if hist.empty:
                continue

            rolling = (
                hist["yield_bu_acre"]
                .shift(1)
                .rolling(3, min_periods=1)
                .mean()
                .iloc[-1]
            )

            prediction_rows.append({
                "county": county,
                "year": prediction_year,
                "rolling_3yr_mean": rolling,
            })

        yield_hist = pd.DataFrame(prediction_rows)

    # ======================================================
    # FEATURE GENERATION
    # ======================================================
    rows = []

    for _, r in yield_hist.iterrows():

        county = r["county"]
        year = int(r["year"])

        nd = ndvi[(ndvi["county"] == county) & (ndvi["year"] == year)]
        wd = wx[(wx["county"] == county) & (wx["year"] == year)]
        storm_year = storm_df[
            (storm_df["county"] == county) &
            (storm_df["year"] == year)
        ]

        if nd.empty or wd.empty:
            continue

        feats = {
            "county": county,
            "year": year,
            "rolling_3yr_mean": r["rolling_3yr_mean"],
        }

        # Include target only for historical
        if mode == "historical":
            feats["yield_bu_acre"] = r["yield_bu_acre"]
            feats["lag1_yield"] = r.get("lag1_yield")

        # ---------------- NDVI ----------------
        nd_cut = nd[cutoff_mask(nd["date"], cutoff_month, cutoff_day)]
        if nd_cut.empty:
            continue

        nd_cut = nd_cut.sort_values("date")
        ndvi_col = "NDVI_smooth" if "NDVI_smooth" in nd_cut.columns else "NDVI"

        peak_row = nd_cut.loc[nd_cut[ndvi_col].idxmax()]
        feats["ndvi_peak"] = float(peak_row[ndvi_col])

        if len(nd_cut) >= 2:
            x = nd_cut["date"].map(pd.Timestamp.toordinal).to_numpy()
            y = nd_cut[ndvi_col].to_numpy()
            feats["ndvi_slope"] = float(np.polyfit(x, y, 1)[0])
        else:
            feats["ndvi_slope"] = np.nan

        # ---------------- WEATHER ----------------
        wd_cut = wd[
            cutoff_mask(wd["date"], cutoff_month, cutoff_day) &
            (wd["date"].dt.month >= 4)
        ]

        if wd_cut.empty:
            continue

        feats["heat_days_gt32"] = int(
            (wd_cut["temperature"] > 29).sum()
        )

        county_temp_mean = float(temp_hist_mean.get(county, 20.0))
        feats["temp_anomaly"] = float(
            wd_cut["temperature"].mean() - county_temp_mean
        )

        rain_sum = float(wd_cut["rain_mm"].sum())

        water_balance = (
            float(wd_cut["water_balance_mm"].sum())
            if "water_balance_mm" in wd_cut.columns
            else 0.0
        )

        feats["net_moisture_stress"] = (
            water_balance / (rain_sum + 1e-6)
        )

        # ---------------- STORM ----------------
        # Unique severe wind days (>= 58 mph)
        # Collapse to daily county max to avoid station density bias

        storms = storm_year[
            (storm_year["datetime"].dt.month >= 4) &
            (
                    (storm_year["datetime"].dt.month < cutoff_month) |
                    (
                            (storm_year["datetime"].dt.month == cutoff_month) &
                            (storm_year["datetime"].dt.day <= cutoff_day)
                    )
            )
            ].copy()

        if storms.empty:
            feats["wind_severe_days_58_cutoff"] = 0
        else:
            storms["date"] = storms["datetime"].dt.date
            daily_max = storms.groupby("date")["wind_mph"].max()

            feats["wind_severe_days_58_cutoff"] = int(
                (daily_max >= 58).sum()
            )
        rows.append(feats)

    feature_df = pd.DataFrame(rows)
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return feature_df