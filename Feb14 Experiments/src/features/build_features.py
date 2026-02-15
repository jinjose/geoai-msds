import pandas as pd
import numpy as np
from features.bio_features import ndvi_features, weather_features


# ============================================================
# SAFE Rolling Trend
# ============================================================
def rolling_trend(series: pd.Series, window: int = 5) -> pd.Series:
    slopes = []

    for i in range(len(series)):
        if i < window:
            slopes.append(np.nan)
        else:
            y = series.iloc[i - window:i]

            if y.isna().any():
                slopes.append(np.nan)
            else:
                x = np.arange(window)
                slopes.append(np.polyfit(x, y, 1)[0])

    return pd.Series(slopes, index=series.index)


import pandas as pd
import numpy as np
from features.bio_features import ndvi_features, weather_features


def build_feature_table(
        yield_df,
        ndvi,
        wx,
        cutoff_month,
        cutoff_day,
        ndvi_hist_mean,
        temp_hist_mean,
):

    # --------------------------------------------------------
    # 1. Compute yield memory CORRECTLY per county
    # --------------------------------------------------------
    yield_hist = yield_df.sort_values(["county", "year"]).copy()

    # Lag
    yield_hist["lag1_yield"] = (
        yield_hist.groupby("county")["yield_bu_acre"].shift(1)
    )

    # Rolling mean (3 years)
    yield_hist["rolling_3yr_mean"] = (
        yield_hist.groupby("county")["yield_bu_acre"]
        .apply(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )

    # Rolling std (5 years)
    yield_hist["yield_std_5yr"] = (
        yield_hist.groupby("county")["yield_bu_acre"]
        .apply(lambda x: x.shift(1).rolling(5, min_periods=5).std())
        .reset_index(level=0, drop=True)
    )

    # Rolling trend (5 years)
    def compute_trend(x: pd.Series, window: int = 5) -> pd.Series:
        x_shift = x.shift(1)
        result = []

        for i in range(len(x_shift)):
            start = i - window
            end = i

            if start < 0:
                result.append(np.nan)
                continue

            y = x_shift.iloc[start:end]

            if y.isna().any() or len(y) != window:
                result.append(np.nan)
                continue

            t = np.arange(window)
            slope = np.polyfit(t, y.values, 1)[0]
            result.append(slope)

        return pd.Series(result, index=x.index)

    # def compute_trend(x):
    #     x_shift = x.shift(1)
    #     trends = []
    #     for i in range(len(x_shift)):
    #         if i < 5:
    #             trends.append(np.nan)
    #         else:
    #             y = x_shift.iloc[i-5:i]
    #             if y.isna().any():
    #                 trends.append(np.nan)
    #             else:
    #                 slope = np.polyfit(range(5), y, 1)[0]
    #                 trends.append(slope)
    #     return pd.Series(trends, index=x.index)

    yield_hist["yield_trend_5yr"] = (
        yield_hist.groupby("county")["yield_bu_acre"]
        .apply(lambda x:
               x.shift(1)
               .rolling(5, min_periods=5)
               .apply(lambda y: np.polyfit(np.arange(len(y)), y, 1)[0],
                      raw=False)
               )
        .reset_index(level=0, drop=True)
    )

    # --------------------------------------------------------
    # 2. Restrict to NDVI/Weather years
    # --------------------------------------------------------
    valid_years = sorted(set(ndvi["year"]).intersection(set(wx["year"])))
    yield_hist = yield_hist[yield_hist["year"].isin(valid_years)]

    rows = []

    # --------------------------------------------------------
    # 3. Build features
    # --------------------------------------------------------
    for _, r in yield_hist.iterrows():

        # require full 5-year history
        if pd.isna(r["yield_std_5yr"]):
            continue

        county = r["county"]
        year = r["year"]

        nd = ndvi[(ndvi["county"] == county) & (ndvi["year"] == year)]
        wd = wx[(wx["county"] == county) & (wx["year"] == year)]

        if nd.empty or wd.empty:
            continue

        feats = {
            "county": county,
            "year": year,
            "yield_bu_acre": r["yield_bu_acre"],
            "lag1_yield": r["lag1_yield"],
            "rolling_3yr_mean": r["rolling_3yr_mean"],
            "yield_std_5yr": r["yield_std_5yr"],
            "yield_trend_5yr": r["yield_trend_5yr"],
        }

        # NDVI
        feats.update(
            ndvi_features(
                nd,
                cutoff_month,
                cutoff_day,
                ndvi_hist_mean.loc[county],
            )
        )

        # Weather
        feats.update(
            weather_features(
                wd,
                cutoff_month,
                cutoff_day,
                temp_hist_mean.loc[county],
            )
        )

        # Interactions
        rain = feats.get("rain_sum_mm", 0)
        heat = feats.get("heat_days_gt32", 0)

        if "water_balance_total_mm" in feats:
            feats["net_moisture_stress"] = (
                feats["water_balance_total_mm"] / (rain + 1e-6)
            )

        feats["heat_rain_stress_idx"] = heat / (rain + 1e-6)

        rows.append(feats)

    feature_df = pd.DataFrame(rows)

    if feature_df.empty:
        return feature_df

    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return feature_df
