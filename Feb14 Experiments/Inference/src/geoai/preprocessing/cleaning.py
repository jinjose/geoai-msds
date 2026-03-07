import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def clean_yield(yield_raw, max_cv_pct=25):
    """
    Standardizes USDA county-level corn yield data.
    Fixes CV filtering issue that was removing older years.

    Parameters
    ----------
    yield_raw : pd.DataFrame
        Raw USDA NASS CSV.
    max_cv_pct : float or None
        Maximum allowable CV percentage.
        Rows with missing CV are retained.

    Returns
    -------
    pd.DataFrame with columns:
        county, year, yield_bu_acre
    """

    df = yield_raw.copy()

    # -----------------------------
    # Standardize core columns
    # -----------------------------
    df["county"] = df["County"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Remove commas in yield values (if any)
    df["yield_bu_acre"] = (
        df["Value"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df["yield_bu_acre"] = pd.to_numeric(df["yield_bu_acre"], errors="coerce")

    # -----------------------------
    # Remove pseudo counties
    # -----------------------------
    df = df[
        ~df["county"]
        .str.upper()
        .str.contains("OTHER COUNTIES", na=False)
    ]

    # -----------------------------
    # Agronomic plausibility filter
    # -----------------------------
    df = df[
        (df["yield_bu_acre"] >= 80) &
        (df["yield_bu_acre"] <= 300)
    ]

    # -----------------------------
    # CV filtering (FIXED LOGIC)
    # -----------------------------
    if "CV (%)" in df.columns:
        df["cv_pct"] = pd.to_numeric(df["CV (%)"], errors="coerce")

        if max_cv_pct is not None:
            # Keep rows where CV is missing OR <= threshold
            df = df[
                (df["cv_pct"].isna()) |
                (df["cv_pct"] <= float(max_cv_pct))
            ]

    # -----------------------------
    # Final cleaning
    # -----------------------------
    df = df.dropna(subset=["county", "year", "yield_bu_acre"])
    df = df.drop_duplicates(subset=["county", "year"])

    print("Yield years after cleaning:",
          sorted(df["year"].unique()))

    return df[["county", "year", "yield_bu_acre"]]



def clean_ndvi(ndvi_raw, min_points_per_county_year=6):

    df = ndvi_raw.copy()

    # Always compute year from date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")

    df = df.dropna(subset=["county_name", "date", "year", "NDVI"])

    df["county"] = df["county_name"].astype(str)

    df = df[(df["NDVI"] >= -0.1) & (df["NDVI"] <= 1.0)]

    # Aggregate duplicates safely
    df = df.groupby(["county", "date", "year"], as_index=False)["NDVI"].mean()

    # Count points per county-year
    counts = df.groupby(["county", "year"]).size()

    keep = counts[counts >= min_points_per_county_year].index

    cleaned = (
        df.set_index(["county", "year"])
          .loc[keep]
          .reset_index()
    )

    print("NDVI years after cleaning:",
          sorted(cleaned["year"].unique()))

    return cleaned


def smooth_county_ndvi(df, window=9, poly=2):
    """
    Applies Savitzky-Golay smoothing to remove satellite noise while preserving peaks.
    """

    def _apply_sg(group):
        group = group.sort_values("date")
        if len(group) >= window:
            group["NDVI_smooth"] = savgol_filter(group["NDVI"], window, poly)
        else:
            group["NDVI_smooth"] = group["NDVI"]
        return group

    return df.groupby(["county", "year"], group_keys=False).apply(_apply_sg)


def clean_weather(
        wx_raw,
        growing_months=(4, 9),
        rainfall_unit="m",
        min_days_per_county_year=150
):
    """
    Standardizes ERA5 weather data.
    Calculates VPD and Water Balance.
    FIXED: Always compute year from date to avoid silent truncation.
    """

    df = wx_raw.copy()

    # -------------------------------------------------------
    # Ensure date is datetime and compute year/month
    # -------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # -------------------------------------------------------
    # Convert numeric columns safely
    # -------------------------------------------------------
    target_cols = ["temperature", "rainfall",
                   "evapotranspiration", "dewpoint_temperature"]

    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------------------------
    # Kelvin to Celsius conversion
    # -------------------------------------------------------
    for col in ["temperature", "dewpoint_temperature"]:
        if col in df.columns and df[col].mean() > 100:
            df[col] = df[col] - 273.15

    # -------------------------------------------------------
    # Rainfall and ET conversion
    # -------------------------------------------------------
    if rainfall_unit == "m":
        df["rain_mm"] = df["rainfall"] * 1000.0
    else:
        df["rain_mm"] = df["rainfall"]

    if "evapotranspiration" in df.columns:
        df["et_mm"] = abs(df["evapotranspiration"]) * 1000.0
        df["water_balance_mm"] = df["rain_mm"] - df["et_mm"]

    # -------------------------------------------------------
    # Vapor Pressure Deficit
    # -------------------------------------------------------
    if "temperature" in df.columns and "dewpoint_temperature" in df.columns:
        es = 0.6108 * np.exp(
            (17.27 * df["temperature"]) /
            (df["temperature"] + 237.3)
        )
        ea = 0.6108 * np.exp(
            (17.27 * df["dewpoint_temperature"]) /
            (df["dewpoint_temperature"] + 237.3)
        )
        df["vpd_kpa"] = es - ea

    # -------------------------------------------------------
    # Drop rows missing key fields
    # -------------------------------------------------------
    df = df.dropna(subset=["county_name", "date", "year"])
    df["county"] = df["county_name"].astype(str)

    # -------------------------------------------------------
    # Growing season filter
    # -------------------------------------------------------
    df = df[df["month"].between(growing_months[0], growing_months[1])]

    # -------------------------------------------------------
    # Ensure sufficient daily coverage
    # -------------------------------------------------------
    if min_days_per_county_year > 0:
        counts = df.groupby(["county", "year"]).size()

        print("\nWeather days per year (avg):")
        print(counts.groupby("year").mean())

        keep = counts[counts >= min_days_per_county_year].index

        df = (
            df.set_index(["county", "year"])
              .loc[keep]
              .reset_index()
        )

    print("Weather years after cleaning:",
          sorted(df["year"].unique()))

    return df



def enforce_intersection(yield_df, ndvi_df, wx_df):
    """
    Ensures data consistency across all three sources.
    """
    counties = set(yield_df["county"]) & set(ndvi_df["county"]) & set(wx_df["county"])
    years = set(yield_df["year"]) & set(ndvi_df["year"]) & set(wx_df["year"])
    return (yield_df[yield_df["county"].isin(counties) & yield_df["year"].isin(years)],
            ndvi_df[ndvi_df["county"].isin(counties) & ndvi_df["year"].isin(years)],
            wx_df[wx_df["county"].isin(counties) & wx_df["year"].isin(years)])