import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


# ============================================================
# YIELD CLEANING
# ============================================================
def clean_yield(yield_raw, max_cv_pct=25, strict=True):

    df = yield_raw.copy()

    df["county"] = (
        df["County"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["year"] = pd.to_numeric(df["Year"], errors="coerce")

    df["yield_bu_acre"] = (
        df["Value"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df["yield_bu_acre"] = pd.to_numeric(df["yield_bu_acre"], errors="coerce")

    df = df[
        ~df["county"]
        .str.upper()
        .str.contains("OTHER COUNTIES", na=False)
    ]

    if strict:
        df = df[
            (df["yield_bu_acre"] >= 80) &
            (df["yield_bu_acre"] <= 300)
        ]

    if strict and "CV (%)" in df.columns:
        df["cv_pct"] = pd.to_numeric(df["CV (%)"], errors="coerce")
        if max_cv_pct is not None:
            df = df[
                (df["cv_pct"].isna()) |
                (df["cv_pct"] <= float(max_cv_pct))
            ]

    df = df.dropna(subset=["county", "year", "yield_bu_acre"])
    df = df.drop_duplicates(subset=["county", "year"])

    return df[["county", "year", "yield_bu_acre"]]


# ============================================================
# NDVI CLEANING
# ============================================================
def clean_ndvi(ndvi_raw, min_points_per_county_year=6, strict=True):

    df = ndvi_raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")

    df = df.dropna(subset=["county_name", "date", "year", "NDVI"])

    df["county"] = (
        df["county_name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[(df["NDVI"] >= -0.1) & (df["NDVI"] <= 1.0)]

    df = df.groupby(["county", "date", "year"], as_index=False)["NDVI"].mean()

    if strict:
        counts = df.groupby(["county", "year"]).size()
        keep = counts[counts >= min_points_per_county_year].index

        df = (
            df.set_index(["county", "year"])
              .loc[keep]
              .reset_index()
        )

    return df


# ============================================================
# NDVI SMOOTHING (UPDATED)
# ============================================================
def smooth_county_ndvi(df, window=9, poly=2):

    if window % 2 == 0:
        window += 1

    def _apply_sg(group):
        group = group.sort_values("date")

        if len(group) >= window:
            group["NDVI_smooth"] = savgol_filter(group["NDVI"], window, poly)
        else:
            group["NDVI_smooth"] = group["NDVI"]

        return group

    return (
        df
        .sort_values(["county", "year", "date"])
        .groupby(["county", "year"], group_keys=False)
        .apply(_apply_sg, include_groups=False)  # <-- FIXED
        .reset_index(drop=True)
    )


# ============================================================
# WEATHER CLEANING
# ============================================================
def clean_weather(
        wx_raw,
        growing_months=(4, 9),
        rainfall_unit="m",
        min_days_per_county_year=150,
        strict=True
):

    df = wx_raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df = df.dropna(subset=["county_name", "date", "year"])

    df["county"] = (
        df["county_name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    numeric_cols = [
        "temperature",
        "rainfall",
        "evapotranspiration",
        "dewpoint_temperature"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Kelvin to Celsius safety check
    for col in ["temperature", "dewpoint_temperature"]:
        if col in df.columns and df[col].mean(skipna=True) > 100:
            df[col] = df[col] - 273.15

    if "rainfall" in df.columns:
        if rainfall_unit == "m":
            df["rain_mm"] = df["rainfall"] * 1000.0
        else:
            df["rain_mm"] = df["rainfall"]
    else:
        df["rain_mm"] = 0.0

    if "evapotranspiration" in df.columns:
        df["et_mm"] = abs(df["evapotranspiration"]) * 1000.0
        df["water_balance_mm"] = df["rain_mm"] - df["et_mm"]
    else:
        df["water_balance_mm"] = 0.0

    df = df[df["month"].between(growing_months[0], growing_months[1])]

    if strict and min_days_per_county_year > 0:
        counts = df.groupby(["county", "year"]).size()
        keep = counts[counts >= min_days_per_county_year].index
        df = (
            df.set_index(["county", "year"])
              .loc[keep]
              .reset_index()
        )

    return df


# ============================================================
# STORM CLEANING
# ============================================================
def clean_storm(storm_raw, strict=True):

    df = storm_raw.copy()

    df["county"] = (
        df["CZ_NAME"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["datetime"] = pd.to_datetime(
        df["BEGIN_DATE_TIME"],
        errors="coerce"
    )

    df["year"] = df["datetime"].dt.year

    df["wind_mph"] = pd.to_numeric(
        df["MAGNITUDE"],
        errors="coerce"
    )

    df = df[
        df["EVENT_TYPE"].isin(
            ["Thunderstorm Wind", "High Wind"]
        )
    ]

    df = df.dropna(subset=["county", "datetime", "wind_mph"])

    if strict:
        df = df[
            (df["wind_mph"] >= 10) &
            (df["wind_mph"] <= 200)
        ]

    df = df.drop_duplicates(
        subset=["county", "datetime", "wind_mph"]
    )

    return df[["county", "datetime", "year", "wind_mph"]]


# ============================================================
# LENIENT DATA CONSISTENCY ENFORCEMENT
# ============================================================
def enforce_intersection_lenient(yield_df, ndvi_df, wx_df):

    yield_df = yield_df.copy()
    ndvi_df = ndvi_df.copy()
    wx_df = wx_df.copy()

    counties = (
        set(yield_df["county"]) &
        set(ndvi_df["county"]) &
        set(wx_df["county"])
    )

    years = (
        set(yield_df["year"]) &
        set(ndvi_df["year"]) &
        set(wx_df["year"])
    )

    yield_df = yield_df[
        yield_df["county"].isin(counties) &
        yield_df["year"].isin(years)
    ]

    ndvi_df = ndvi_df[
        ndvi_df["county"].isin(counties) &
        ndvi_df["year"].isin(years)
    ]

    wx_df = wx_df[
        wx_df["county"].isin(counties) &
        wx_df["year"].isin(years)
    ]

    return yield_df, ndvi_df, wx_df