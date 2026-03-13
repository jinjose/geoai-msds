import pandas as pd
import numpy as np

import sys
from pathlib import Path

# assumes geoai_local/
#   ingestion_container/
#   feature_container/

ROOT = Path(__file__).resolve().parents[1]   # geoai_local
sys.path.insert(0, str(ROOT / "feature_container"))


from .bio_features import ndvi_features, weather_features


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

def _safe_memory_features(yh_past: pd.DataFrame) -> dict:
    """
    Compute yield memory features using available history.
    Fallbacks prevent empty feature tables in forecast mode.
    """
    s = yh_past.sort_values("year")["yield_bu_acre"].astype(float)

    out = {
        "lag1_yield": float(s.iloc[-1]),
        "rolling_3yr_mean": float(s.tail(3).mean()) if len(s) >= 3 else float(s.mean()),
        "yield_std_5yr": float(s.tail(5).std(ddof=1)) if len(s) >= 3 else float(s.std(ddof=1)) if len(s) >= 2 else 0.0,
        "yield_trend_5yr": 0.0,
    }

    # trend requires >=3 points
    if len(s) >= 3:
        y = s.tail(min(5, len(s))).values
        x = np.arange(len(y))
        out["yield_trend_5yr"] = float(np.polyfit(x, y, 1)[0])

    return out

def build_feature_table(
    yield_df,
    ndvi,
    wx,
    cutoff_month,
    cutoff_day,
    ndvi_hist_mean,
    temp_hist_mean,
    target_years=None,
    storm_df=None,
):
    """
    Old logic + permanent fixes:
      - County normalization across all datasets (fixes O'Brien and punctuation/spacing issues)
      - Still uses NDVI ∩ WX for county universe (training-style; prevents NaNs)
      - Restores interaction/derived features:
          net_moisture_stress = water_balance_total_mm / (rain_sum_mm + 1e-6)
          heat_rain_stress_idx = heat_days_gt32 / (rain_sum_mm + 1e-6)
      - Storm is optional; if missing for a county it becomes 0 (no storm days)
    """
    import pandas as pd
    import numpy as np
    import re
    import unicodedata

    if target_years is not None:
        target_years = set([int(y) for y in target_years])

    # ----------------------------
    # Canonical county normalizer
    # ----------------------------
    def norm_county(x: str) -> str:
        s = "" if x is None else str(x)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = s.lower().strip()
        s = re.sub(r"\s+county$", "", s)
        s = s.replace("'", " ").replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()
        # ensure O'Brien matches across sources
        if s == "obrien":
            s = "o brien"
        return s

    # ----------------------------
    # Defensive copies + type fixes
    # ----------------------------
    ndvi = ndvi.copy()
    wx = wx.copy()
    yield_df = yield_df.copy()

    # Force weather date to datetime BEFORE feature logic
    if "date" in wx.columns:
        wx["date"] = pd.to_datetime(wx["date"], errors="coerce")
        wx = wx.dropna(subset=["date"])

    # Normalize years
    ndvi["year"] = pd.to_numeric(ndvi["year"], errors="coerce").astype("Int64")
    wx["year"] = pd.to_numeric(wx["year"], errors="coerce").astype("Int64")
    yield_df["year"] = pd.to_numeric(yield_df["year"], errors="coerce").astype("Int64")

    # Normalize counties everywhere
    if "county" in ndvi.columns:
        ndvi["county"] = ndvi["county"].astype(str).map(norm_county)
    if "county" in wx.columns:
        wx["county"] = wx["county"].astype(str).map(norm_county)
    if "county" in yield_df.columns:
        yield_df["county"] = yield_df["county"].astype(str).map(norm_county)

    if storm_df is not None and not storm_df.empty:
        storm_df = storm_df.copy()
        if "county" in storm_df.columns:
            storm_df["county"] = storm_df["county"].astype(str).map(norm_county)

    # ----------------------------
    # 1) Yield history memory features
    # ----------------------------
    yield_hist = yield_df.sort_values(["county", "year"]).copy()

    # Keep your richer memory features (safe)
    yield_hist["lag1_yield"] = yield_hist.groupby("county")["yield_bu_acre"].shift(1)

    yield_hist["rolling_3yr_mean"] = (
        yield_hist.groupby("county")["yield_bu_acre"]
        .apply(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )

    yield_hist["yield_std_5yr"] = (
        yield_hist.groupby("county")["yield_bu_acre"]
        .apply(lambda x: x.shift(1).rolling(5, min_periods=5).std())
        .reset_index(level=0, drop=True)
    )

    yield_hist["yield_trend_5yr"] = (
        yield_hist.groupby("county")["yield_bu_acre"]
        .apply(
            lambda x: x.shift(1)
            .rolling(5, min_periods=5)
            .apply(lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=False)
        )
        .reset_index(level=0, drop=True)
    )

    # ----------------------------
    # 2) Candidate years + counties from NDVI/WX intersection (training-style)
    # ----------------------------
    valid_years = sorted(set(ndvi["year"].dropna().astype(int)).intersection(set(wx["year"].dropna().astype(int))))
    if target_years is not None:
        valid_years = [y for y in valid_years if y in target_years]

    counties = sorted(set(ndvi["county"].dropna()).intersection(set(wx["county"].dropna())))
    yhist_by_county = {c: yield_hist[yield_hist["county"] == c].sort_values("year") for c in counties}

    rows = []

    # ----------------------------
    # 3) Build per (county, year)
    # ----------------------------
    for county in counties:
        yh = yhist_by_county.get(county)
        if yh is None or yh.empty:
            continue

        for year in valid_years:
            nd = ndvi[(ndvi["county"] == county) & (ndvi["year"].astype(int) == int(year))]
            wd = wx[(wx["county"] == county) & (wx["year"].astype(int) == int(year))]
            if nd.empty or wd.empty:
                continue

            yh_past = yh[yh["year"].astype(int) <= (int(year) - 1)]
            yh_past = yh_past.dropna(subset=["yield_bu_acre"])
            if yh_past.empty:
                continue

            mem = _safe_memory_features(yh_past)

            # Actual yield if available (backtests); NaN for forecast year
            y_actual = np.nan
            if (yh["year"].astype(int) == int(year)).any():
                y_actual = float(yh.loc[yh["year"].astype(int) == int(year), "yield_bu_acre"].iloc[0])

            feats = {
                "county": county,
                "year": int(year),
                "yield_bu_acre": y_actual,
                **mem,
            }

            # NDVI
            feats.update(
                ndvi_features(
                    nd,
                    cutoff_month,
                    cutoff_day,
                    ndvi_hist_mean.loc[county] if (ndvi_hist_mean is not None and county in ndvi_hist_mean.index) else None,
                )
            )

            # Weather
            feats.update(
                weather_features(
                    wd,
                    cutoff_month,
                    cutoff_day,
                    temp_hist_mean.loc[county] if (temp_hist_mean is not None and county in temp_hist_mean.index) else None,
                )
            )

            # ----------------------------
            # Interaction features (RESTORE EXACT OLD LOGIC)
            # ----------------------------
            rain = float(feats.get("rain_sum_mm", 0.0) or 0.0)
            heat = float(feats.get("heat_days_gt32", 0.0) or 0.0)

            # net_moisture_stress expected by schema (your original definition)
            if "water_balance_total_mm" in feats and feats["water_balance_total_mm"] is not None:
                wb = float(feats["water_balance_total_mm"])
                feats["net_moisture_stress"] = wb / (rain + 1e-6)
            else:
                # fail-safe: still create the column so freeze doesn't crash
                feats["net_moisture_stress"] = np.nan

            feats["heat_rain_stress_idx"] = heat / (rain + 1e-6)

            rows.append(feats)

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        return feature_df

    # ----------------------------
    # 4) Storm (optional, left-join; missing -> 0)
    # ----------------------------
    if storm_df is not None and not storm_df.empty:
        storm_feats = compute_wind_severe_days_58_cutoff(storm_df, cutoff_month, cutoff_day)
        feature_df = feature_df.merge(storm_feats, on=["county", "year"], how="left")

    feature_df["wind_severe_days_58_cutoff"] = feature_df.get("wind_severe_days_58_cutoff", 0)
    feature_df["wind_severe_days_58_cutoff"] = feature_df["wind_severe_days_58_cutoff"].fillna(0).astype(int)

    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df = feature_df.drop_duplicates(subset=["county", "year"], keep="last")

    return feature_df
#MOST OF THE FUNCTION IS WORKING 
# def build_feature_table(
#         yield_df,
#         ndvi,
#         wx,
#         cutoff_month,
#         cutoff_day,
#         ndvi_hist_mean,
#         temp_hist_mean,
#         target_years=None,
#         storm_df=None,
# ):
#     """
#     Build a feature table for training/backtesting AND forecasting.

#     Forecasting mode example:
#       - yield history exists through 2024
#       - NDVI/Weather exists for 2025 up to cutoff
#       - We export features for year=2025 even if yield_bu_acre for 2025 is not available.

#     target_years:
#       If provided, builds rows only for these years (ideal for inference exports).
#     """
#     if target_years is not None:
#         target_years = set([int(y) for y in target_years])
    
#     # normalize year types (avoid float vs int mismatch)
#     ndvi = ndvi.copy()
#     wx = wx.copy()
#     # FORCE weather date to datetime BEFORE any feature logic
#     if "date" in wx.columns:
#         wx["date"] = pd.to_datetime(wx["date"], errors="coerce")
#         wx = wx.dropna(subset=["date"])

#     ndvi["year"] = pd.to_numeric(ndvi["year"], errors="coerce").astype("Int64").astype(int)
#     wx["year"]   = pd.to_numeric(wx["year"], errors="coerce").astype("Int64").astype(int)
    

#     # --------------------------------------------------------
#     # 1. Yield history memory features per county
#     # --------------------------------------------------------
#     yield_hist = yield_df.sort_values(["county", "year"]).copy()

#     yield_hist["lag1_yield"] = yield_hist.groupby("county")["yield_bu_acre"].shift(1)

#     yield_hist["rolling_3yr_mean"] = (
#         yield_hist.groupby("county")["yield_bu_acre"]
#         .apply(lambda x: x.shift(1).rolling(3, min_periods=3).mean())
#         .reset_index(level=0, drop=True)
#     )

#     yield_hist["yield_std_5yr"] = (
#         yield_hist.groupby("county")["yield_bu_acre"]
#         .apply(lambda x: x.shift(1).rolling(5, min_periods=5).std())
#         .reset_index(level=0, drop=True)
#     )

#     yield_hist["yield_trend_5yr"] = (
#         yield_hist.groupby("county")["yield_bu_acre"]
#         .apply(lambda x:
#                x.shift(1)
#                .rolling(5, min_periods=5)
#                .apply(lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=False)
#                )
#         .reset_index(level=0, drop=True)
#     )

#     # --------------------------------------------------------
#     # 2. Candidate years from NDVI/Weather intersection
#     # --------------------------------------------------------
#     valid_years = sorted(set(ndvi["year"]).intersection(set(wx["year"])))
#     if target_years is not None:
#         valid_years = [y for y in valid_years if y in target_years]

#     counties = sorted(set(ndvi["county"]).intersection(set(wx["county"])))
#     yhist_by_county = {c: yield_hist[yield_hist["county"] == c].sort_values("year") for c in counties}

#     rows = []

#     # --------------------------------------------------------
#     # 3. Build per (county, year)
#     # --------------------------------------------------------
#     for county in counties:
#         yh = yhist_by_county.get(county)
#         if yh is None or yh.empty:
#             continue

#         for year in valid_years:
#             nd = ndvi[(ndvi["county"] == county) & (ndvi["year"] == year)]
#             wd = wx[(wx["county"] == county) & (wx["year"] == year)]
#             if nd.empty or wd.empty:
#                 continue

#             # Use yield history up to year-1 for memory features
#             yh_past = yh[yh["year"] <= (year - 1)]
#             if yh_past.empty:
#                 continue
#             last = yh_past.iloc[-1]

#             # Require enough history (5-year memory available as of year-1)
#             # if pd.isna(last.get("yield_std_5yr")) or pd.isna(last.get("yield_trend_5yr")):
#             #     continue
            
#             mem = _safe_memory_features(yh_past)

#             # Actual yield if available (for evaluation/backtests)
#             y_actual = np.nan
#             if (yh["year"] == year).any():
#                 y_actual = float(yh[yh["year"] == year]["yield_bu_acre"].iloc[0])

#             feats = {
#                 "county": county,
#                 "year": int(year),
#                 "yield_bu_acre": y_actual,
#                 # lag1 for forecast year = last available actual yield (year-1)
#                 **mem
#             }

#             # NDVI
#             feats.update(
#                 ndvi_features(
#                     nd,
#                     cutoff_month,
#                     cutoff_day,
#                     ndvi_hist_mean.loc[county] if (ndvi_hist_mean is not None and county in ndvi_hist_mean.index) else None,
#                 )
#             )

#             # Weather
#             feats.update(
#                 weather_features(
#                     wd,
#                     cutoff_month,
#                     cutoff_day,
#                     temp_hist_mean.loc[county] if (temp_hist_mean is not None and county in temp_hist_mean.index) else None,
#                 )
#             )

#             # Interactions
#             rain = feats.get("rain_sum_mm", 0)
#             heat = feats.get("heat_days_gt32", 0)

#             if "water_balance_total_mm" in feats:
#                 feats["net_moisture_stress"] = feats["water_balance_total_mm"] / (rain + 1e-6)

#             feats["heat_rain_stress_idx"] = heat / (rain + 1e-6)

#             rows.append(feats)

#     feature_df = pd.DataFrame(rows)
#     if feature_df.empty:
#         return feature_df


#     # --------------------------------------------------------
#     # 4. Optional storm/wind features
#     # --------------------------------------------------------
#     if storm_df is not None and not storm_df.empty:
#         storm_feats = compute_wind_severe_days_58_cutoff(storm_df, cutoff_month, cutoff_day)
#         feature_df = feature_df.merge(storm_feats, on=["county","year"], how="left")
#     if "wind_severe_days_58_cutoff" in feature_df.columns:
#         feature_df["wind_severe_days_58_cutoff"] = feature_df["wind_severe_days_58_cutoff"].fillna(0).astype(int)

#     feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     # Final safety: ensure exactly 1 row per county-year
#     # keep="last" is fine because rows are appended deterministically; you can use "first" too.
#     feature_df = feature_df.drop_duplicates(subset=["county", "year"], keep="last")

#     return feature_df

def compute_wind_severe_days_58_cutoff(storm_df: pd.DataFrame, cutoff_month: int, cutoff_day: int) -> pd.DataFrame:
    """Per-county per-year count of days with max wind >= 58mph up to cutoff date."""
    if storm_df is None or storm_df.empty:
        return pd.DataFrame(columns=["county", "year", "wind_severe_days_58_cutoff"])

    df = storm_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year.astype("int32")

    daily = df.groupby(["county", "year", "date"], as_index=False)["wind_mph"].max()

    def _count(g: pd.DataFrame) -> int:
        y = int(g["year"].iloc[0])
        cutoff = pd.Timestamp(year=y, month=cutoff_month, day=cutoff_day).date()
        return int((g.loc[g["date"] <= cutoff, "wind_mph"] >= 58).sum())

    out = daily.groupby(["county", "year"], as_index=False).apply(_count)
    out = out.rename(columns={None: "wind_severe_days_58_cutoff"})
    return out

##MOST OF THE FEATURES ARE WORKING 
# def compute_wind_severe_days_58_cutoff(storm_df: pd.DataFrame, cutoff_month: int, cutoff_day: int) -> pd.DataFrame:
#     """Compute per-county per-year storm wind severe day counts (>=58mph) up to cutoff date."""
#     if storm_df is None or storm_df.empty:
#         return pd.DataFrame(columns=["county","year","wind_severe_days_58_cutoff"])

#     df = storm_df.copy()
#     # Ensure datetime exists and is proper datetime
#     df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
#     df = df.dropna(subset=["datetime"])

#     # Derive date + year from datetime
#     df["date"] = df["datetime"].dt.date
#     df["year"] = df["datetime"].dt.year.astype("Int64")
#     # daily county max wind
#     daily = df.groupby(["county","year","date"], as_index=False)["wind_mph"].max()

#     def _count(g: pd.DataFrame) -> int:
#         y = int(g["year"].iloc[0])
#         cutoff = pd.Timestamp(year=y, month=cutoff_month, day=cutoff_day).date()
#         return int((g[g["date"] <= cutoff]["wind_mph"] >= 58).sum())

#     out = (
#         daily.groupby(["county","year"], as_index=False)
#         .apply(_count)
#         .rename(columns={None:"wind_severe_days_58_cutoff"})
#     )

#     # pandas groupby apply produces a column named 0 sometimes; normalize
#     if 0 in out.columns and "wind_severe_days_58_cutoff" not in out.columns:
#         out = out.rename(columns={0:"wind_severe_days_58_cutoff"})
#     return out[["county","year","wind_severe_days_58_cutoff"]]
