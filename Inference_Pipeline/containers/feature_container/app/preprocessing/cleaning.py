import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os, sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]   # geoai_local
sys.path.insert(0, str(ROOT / "feature_container"))

from app.config import FORECAST_CUTOFFS
from app.config import MIN_NDVI_POINTS_PER_YEAR 
from app.config import FORECAST_CUTOFFS, MIN_NDVI_POINTS_PER_YEAR  # :contentReference[oaicite:4]{index=4}


import os
import pandas as pd


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
    # Standardize core columns (ROBUST)
    # -----------------------------
    # county
    if "county" in df.columns:
        df["county"] = df["county"].astype(str).str.strip()
    elif "County" in df.columns:
        df["county"] = df["County"].astype(str).str.strip()
    else:
        raise KeyError(f"Yield data missing county column. Columns: {list(df.columns)}")

    # year
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    elif "Year" in df.columns:
        df["year"] = pd.to_numeric(df["Year"], errors="coerce")
    else:
        raise KeyError(f"Yield data missing year column. Columns: {list(df.columns)}")

    # yield value
    if "yield_bu_acre" not in df.columns:
        if "Value" not in df.columns:
            raise KeyError(f"Yield data missing Value/yield_bu_acre column. Columns: {list(df.columns)}")

        df["yield_bu_acre"] = (
            df["Value"].astype(str).str.replace(",", "", regex=False)
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

def apply_ndvi_cutoff_fallback(ndvi_df: pd.DataFrame, predict_year: int, feature_season: str) -> pd.DataFrame:
    """
    If NDVI for predict_year has insufficient rows per county up to the cutoff,
    fall back to (predict_year-1) NDVI up to the same cutoff, mapped into predict_year.

    Works for daily OR yearly NDVI, and uses FORECAST_CUTOFFS from config.py.
    Expects columns: county, date, year, NDVI (and optionally NDVI_smooth later).
    """
    if ndvi_df is None or ndvi_df.empty:
        return ndvi_df

    season = str(feature_season).lower().strip()
    if season not in FORECAST_CUTOFFS:
        raise ValueError(f"Unknown FEATURE_SEASON={season}. Expected one of: {list(FORECAST_CUTOFFS.keys())}")

    ndvi_gran = os.getenv("NDVI_GRANULARITY", "daily").lower().strip()
    min_pts = 1 if ndvi_gran == "yearly" else int(MIN_NDVI_POINTS_PER_YEAR)

    df = ndvi_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["county", "date", "NDVI"])
    df["year"] = df["date"].dt.year.astype(int)

    cutoff_m, cutoff_d = FORECAST_CUTOFFS[season]
    cutoff_py = pd.Timestamp(year=int(predict_year), month=cutoff_m, day=cutoff_d)
    cutoff_fb = pd.Timestamp(year=int(predict_year - 1), month=cutoff_m, day=cutoff_d)

    # predict year slice up to cutoff
    py = df[(df["year"] == int(predict_year)) & (df["date"] <= cutoff_py)].copy()

    # counties that are "OK" in predict_year
    py_counts = py.groupby("county").size()
    ok = set(py_counts[py_counts >= min_pts].index)

    all_counties = set(df["county"].unique())
    need = all_counties - ok

    if not need:
        # predict_year already sufficient
        print(f"NDVI fallback AUTO: predict_year {predict_year} sufficient (min_pts={min_pts})")
        return df

    # fallback year slice up to cutoff for missing/insufficient counties
    fb = df[(df["year"] == int(predict_year - 1)) & (df["date"] <= cutoff_fb) & (df["county"].isin(need))].copy()
    if fb.empty:
        print(f"WARNING: NDVI fallback AUTO: no fallback data for year={predict_year-1}. Proceeding without fallback.")
        return df

    # map fallback into predict_year
    fb["date"] = fb["date"].apply(lambda d: d.replace(year=int(predict_year)))
    fb["year"] = int(predict_year)

    # IMPORTANT:
    # - we want predict_year to exist for downstream year intersection
    # - so we keep ALL historical years, plus "filled" predict_year rows
    out = pd.concat([df, fb], ignore_index=True)

    print(f"NDVI fallback AUTO: filled {len(need)} counties from {predict_year-1} into {predict_year} (min_pts={min_pts})")
    return out

def clean_ndvi(ndvi_raw, min_points_per_county_year=6):
    if ndvi_raw is None or ndvi_raw.empty:
        print("NDVI years after cleaning: [] (empty input)")
        return pd.DataFrame(columns=["county", "date", "year", "NDVI"])

    if "date" not in ndvi_raw.columns or "NDVI" not in ndvi_raw.columns:
        raise ValueError(f"NDVI data missing required columns. cols={list(ndvi_raw.columns)}")

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

    # --- IMPORTANT: yearly NDVI often has 1 record per county-year ---
    ndvi_gran = os.getenv("NDVI_GRANULARITY", "yearly").lower().strip()
    if min_points_per_county_year is None:
        # yearly NDVI often has 1 row per county-year
        min_points_per_county_year = 1 if ndvi_gran == "yearly" else int(MIN_NDVI_POINTS_PER_YEAR)

    if ndvi_gran == "yearly":
        min_points_per_county_year = 1

    counts = df.groupby(["county", "year"]).size()
    keep = counts[counts >= min_points_per_county_year].index

    cleaned = (
        df.set_index(["county", "year"])
          .loc[keep]
          .reset_index()
    )

    # --- NDVI fallback for forecast year (2025 uses 2024) ---
    predict_year = int(os.getenv("PREDICT_YEAR", "2025"))
    force_fallback = os.getenv("FORCE_NDVI_FALLBACK", "true").lower() == "true"
    fallback_year = int(os.getenv("NDVI_FALLBACK_YEAR", str(predict_year - 1)))

    if force_fallback:
        if fallback_year in cleaned["year"].unique():
            src = cleaned[cleaned["year"] == fallback_year].copy()
            src["year"] = predict_year
            # shift date year so cutoff logic behaves consistently
            src["date"] = src["date"].apply(lambda d: d.replace(year=predict_year))
            cleaned = pd.concat(
                [cleaned[cleaned["year"] != predict_year], src],
                ignore_index=True
            )
            print(f"NDVI fallback applied: using {fallback_year} NDVI as {predict_year}")
        else:
            print(f"WARNING: NDVI fallback_year {fallback_year} not present after cleaning; cannot fallback.")

    print("NDVI years after cleaning:", sorted(cleaned["year"].unique()))
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
    min_days_per_county_year=150,
):
    """
    Cleans ERA5 data for inference + training alignment.

    TRAINING-ALIGNED WINDOWING:
      - For CUTOFF models (jun01/jul01/jul15/aug01/aug15):
          Weather window is growing season start (Apr 1 by default) up to cutoff date (inclusive).
          This matches training logic that used: month>=4 AND date<=cutoff.

      - For YEARLY granularity:
          No daily windowing; just normalize numeric fields and county/year.

    Also:
      - Parses date ONCE (no dayfirst)
      - Uses dynamic coverage threshold so cutoff windows don't get wiped by min_days=150
    """
    import os
    import numpy as np
    import pandas as pd
    from app.config import FORECAST_CUTOFFS

    if wx_raw is None or wx_raw.empty:
        print("Weather years after cleaning: [] (empty input)")
        return pd.DataFrame()

    df = wx_raw.copy()

    feature_season = os.getenv("FEATURE_SEASON", "").lower().strip()
    predict_year = int(os.getenv("PREDICT_YEAR", "2025"))
    era5_gran = os.getenv("ERA5_GRANULARITY", "yearly").lower().strip()

    # -------------------------
    # Common: numeric coercions
    # -------------------------
    target_cols = ["temperature", "rainfall", "evapotranspiration", "dewpoint_temperature"]
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Kelvin to Celsius
    for col in ["temperature", "dewpoint_temperature"]:
        if col in df.columns and pd.notna(df[col]).any() and df[col].mean() > 100:
            df[col] = df[col] - 273.15

    # Rainfall conversion
    if "rainfall" in df.columns:
        df["rain_mm"] = df["rainfall"] * 1000.0 if rainfall_unit == "m" else df["rainfall"]

    # ET conversion + water balance
    if "evapotranspiration" in df.columns:
        df["et_mm"] = abs(df["evapotranspiration"]) * 1000.0
        if "rain_mm" in df.columns:
            df["water_balance_mm"] = df["rain_mm"] - df["et_mm"]

    # VPD if possible
    if "temperature" in df.columns and "dewpoint_temperature" in df.columns:
        es = 0.6108 * np.exp((17.27 * df["temperature"]) / (df["temperature"] + 237.3))
        ea = 0.6108 * np.exp((17.27 * df["dewpoint_temperature"]) / (df["dewpoint_temperature"] + 237.3))
        df["vpd_kpa"] = es - ea

    # -------------------------
    # YEARLY PATH
    # -------------------------
    if era5_gran == "yearly":
        # Normalize county
        if "county_name" in df.columns:
            df["county"] = df["county_name"].astype(str)
        elif "county" in df.columns:
            df["county"] = df["county"].astype(str)
        else:
            raise ValueError("ERA5 yearly data must contain county_name or county")

        # Ensure year
        if "year" not in df.columns:
            raise ValueError("ERA5 yearly data must contain a 'year' column")

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["county", "year"])

        print("Weather years after cleaning (YEARLY):", sorted(df["year"].dropna().unique()))
        return df

    # -------------------------
    # DAILY PATH (training-aligned cutoff)
    # -------------------------
    if "date" not in df.columns:
        raise ValueError("ERA5 daily data must contain a 'date' column")

    # Parse date ONCE
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # County
    if "county_name" in df.columns:
        df["county"] = df["county_name"].astype(str)
    elif "county" in df.columns:
        df["county"] = df["county"].astype(str)
    else:
        raise ValueError("ERA5 daily data must contain county_name or county")

    # Apply training-aligned cutoff window: Apr 1 -> cutoff
    if feature_season not in FORECAST_CUTOFFS:
        raise ValueError(f"Unknown FEATURE_SEASON: {feature_season}")

    cutoff_month, cutoff_day = FORECAST_CUTOFFS[feature_season]

    start = pd.Timestamp(f"{predict_year}-{growing_months[0]:02d}-01")  # e.g. Apr 1
    end = pd.Timestamp(f"{predict_year}-{cutoff_month:02d}-{cutoff_day:02d}")

    # growing season months + cutoff
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    # derive partitions
    df["year"] = df["date"].dt.year.astype("int32")
    df["month"] = df["date"].dt.month.astype("int32")
    df["day"] = df["date"].dt.day.astype("int32")

    # drop bad rows
    df = df.dropna(subset=["county", "date", "year"])

    # Dynamic coverage threshold (so jun01 doesn't require 150)
    if min_days_per_county_year > 0:
        counts = df.groupby(["county", "year"]).size()

        print("\nWeather days per year (avg):")
        print(counts.groupby("year").mean())

        expected_days = int((end - start).days + 1)
        dynamic_min = min(min_days_per_county_year, int(0.9 * expected_days))

        max_cov = int(counts.max()) if len(counts) else 0
        if max_cov < dynamic_min:
            print(
                f"WARNING: max days/county-year={max_cov} < dynamic_min_days={dynamic_min} "
                f"(expected_days={expected_days}). Skipping coverage filter."
            )
        else:
            keep = counts[counts >= dynamic_min].index
            df = df.set_index(["county", "year"]).loc[keep].reset_index()

    print("Weather years after cleaning (DAILY):", sorted(df["year"].unique()))
    print("Weather date range:", df["date"].min(), "to", df["date"].max())
    print("Weather unique dates:", df["date"].nunique())

    return df

## WORKING ONE , Before the cleanup 
# def clean_weather(
#     wx_raw,
#     growing_months=(4, 9),
#     rainfall_unit="m",
#     min_days_per_county_year=150
# ):
#     """
#     Cleans ERA5 data for either:
#       - daily granularity (expects date column; applies growing season + min days)
#       - yearly granularity (expects year column; NO daily filters)
#     """
#     if wx_raw is None or wx_raw.empty:
#         print("Weather years after cleaning: [] (empty input)")
#         return pd.DataFrame()   
     
#     df = wx_raw.copy()

#     feature_season = os.getenv("FEATURE_SEASON", "").lower().strip()
#     predict_year = int(os.getenv("PREDICT_YEAR"))

#     if feature_season not in FORECAST_CUTOFFS:
#         raise ValueError(f"Unknown FEATURE_SEASON: {feature_season}")

#     cutoff_month, cutoff_day = FORECAST_CUTOFFS[feature_season]

#     start = pd.Timestamp(f"{predict_year}-01-01")
#     end = pd.Timestamp(f"{predict_year}-{cutoff_month:02d}-{cutoff_day:02d}")

#     # Ensure date column is datetime
#     df["date"] = pd.to_datetime(df["date"], errors="coerce")
#     df = df.dropna(subset=["date"])

#     # Apply window
#     df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

#     # Recompute year/month/day from actual date (never trust stored month)
#     df["year"] = df["date"].dt.year.astype("int32")
#     df["month"] = df["date"].dt.month.astype("int32")
#     df["day"] = df["date"].dt.day.astype("int32")

#     # feature_season = os.getenv("FEATURE_SEASON", "").lower()
#     # cutoff_month = {"jun01": 6, "jul01": 7, "jul15": 7, "aug01": 8, "aug15": 8}.get(feature_season)

#     # if cutoff_month:
#     #     df = df[df["month"].between(1, cutoff_month)]
#     # else:
#     #     df = df[df["month"].between(growing_months[0], growing_months[1])]


#     era5_gran = os.getenv("ERA5_GRANULARITY", "yearly").lower()

#     # -------------------------
#     # Common numeric coercions
#     # -------------------------
#     target_cols = ["temperature", "rainfall", "evapotranspiration", "dewpoint_temperature"]
#     for col in target_cols:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")

#     # Kelvin to Celsius if needed
#     for col in ["temperature", "dewpoint_temperature"]:
#         if col in df.columns and pd.notna(df[col]).any() and df[col].mean() > 100:
#             df[col] = df[col] - 273.15

#     # Rainfall conversion
#     if "rainfall" in df.columns:
#         df["rain_mm"] = df["rainfall"] * 1000.0 if rainfall_unit == "m" else df["rainfall"]

#     # ET conversion
#     if "evapotranspiration" in df.columns:
#         df["et_mm"] = abs(df["evapotranspiration"]) * 1000.0
#         if "rain_mm" in df.columns:
#             df["water_balance_mm"] = df["rain_mm"] - df["et_mm"]

#     # VPD (works for both daily/yearly if temp+dewpoint exist)
#     if "temperature" in df.columns and "dewpoint_temperature" in df.columns:
#         es = 0.6108 * np.exp((17.27 * df["temperature"]) / (df["temperature"] + 237.3))
#         ea = 0.6108 * np.exp((17.27 * df["dewpoint_temperature"]) / (df["dewpoint_temperature"] + 237.3))
#         df["vpd_kpa"] = es - ea

#     # -------------------------
#     # YEARLY PATH (your 2013–2024 CSVs)
#     # -------------------------
#     if era5_gran == "yearly":
#         # Normalize county name
#         if "county_name" in df.columns:
#             df["county"] = df["county_name"].astype(str)
#         elif "county" in df.columns:
#             df["county"] = df["county"].astype(str)
#         else:
#             raise ValueError("ERA5 yearly data must contain county_name or county")

#         # Ensure year exists
#         if "year" not in df.columns:
#             # If file doesn't have year, try infer from filename earlier or caller should add it
#             raise ValueError("ERA5 yearly data must contain a 'year' column")

#         df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
#         df = df.dropna(subset=["county", "year"])

#         print("Weather years after cleaning (YEARLY):", sorted(df["year"].unique()))
#         print("Weather date range:", df["date"].min(), "to", df["date"].max())
        
#         return df

#     # -------------------------
#     # DAILY PATH (2025 daily parquet, etc.)
#     # -------------------------
#     # Ensure date exists and compute year/month
#     if "date" not in df.columns:
#         raise ValueError("ERA5 daily data must contain a 'date' column")

#     df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
#     df["year"] = df["date"].dt.year
#     df["month"] = df["date"].dt.month

#     # Drop rows missing key fields
#     if "county_name" in df.columns:
#         df["county"] = df["county_name"].astype(str)
#     elif "county" in df.columns:
#         df["county"] = df["county"].astype(str)
#     else:
#         raise ValueError("ERA5 daily data must contain county_name or county")

#     df = df.dropna(subset=["county", "date", "year"])

#     # Growing season filter
#     df = df[df["month"].between(growing_months[0], growing_months[1])]

    
#     # # Ensure sufficient daily coverage

#     if min_days_per_county_year > 0:
#         counts = df.groupby(["county", "year"]).size()

#         print("\nWeather days per year (avg):")
#         print(counts.groupby("year").mean())

#         # If data is not truly daily, this filter will wipe everything.
#         # So we only apply it if at least one county-year meets the threshold.
#         max_cov = counts.max() if len(counts) else 0

#         if max_cov < min_days_per_county_year:
#             print(
#                 f"WARNING: max days/county-year={max_cov} < min_days_per_county_year={min_days_per_county_year}. "
#                 f"Skipping coverage filter to avoid empty weather table."
#             )
#         else:
#             keep = counts[counts >= min_days_per_county_year].index
#             df = df.set_index(["county", "year"]).loc[keep].reset_index()

#     # if min_days_per_county_year > 0:
#     #     counts = df.groupby(["county", "year"]).size()
#     #     print("\nWeather days per year (avg):")
#     #     print(counts.groupby("year").mean())

#     #     keep = counts[counts >= min_days_per_county_year].index
#     #     df = df.set_index(["county", "year"]).loc[keep].reset_index()

#     print("Weather years after cleaning (DAILY):", sorted(df["year"].unique()))
#     return df



def enforce_intersection(yield_df, ndvi_df, wx_df, predict_year: int, mode: str = "forecast"):
    """
    mode:
      - "forecast": keep predictors for predict_year; keep yield history up to predict_year-1
      - "train":    intersect on county+year across all three
    """
    if mode == "train":
        counties = set(yield_df["county"]) & set(ndvi_df["county"]) & set(wx_df["county"])
        years = set(yield_df["year"]) & set(ndvi_df["year"]) & set(wx_df["year"])
        return (
            yield_df[yield_df["county"].isin(counties) & yield_df["year"].isin(years)],
            ndvi_df[ndvi_df["county"].isin(counties) & ndvi_df["year"].isin(years)],
            wx_df[wx_df["county"].isin(counties) & wx_df["year"].isin(years)],
        )

    # ---------- forecast ----------
    # predictors must be ONLY predict_year
    nd = ndvi_df[ndvi_df["year"] == predict_year].copy()
    wx = wx_df[wx_df["year"] == predict_year].copy()
    if nd.empty:
        fb = int(os.getenv("NDVI_FALLBACK_YEAR", str(predict_year - 1)))
        print(f"WARNING: NDVI missing for {predict_year}. Falling back to {fb}.")
        nd = ndvi_df[ndvi_df["year"] == fb].copy()
        if not nd.empty:
            nd["year"] = predict_year
            if "date" in nd.columns:
                nd["date"] = pd.to_datetime(nd["date"], errors="coerce").apply(lambda d: d.replace(year=predict_year))
              
    # yield is historical only (up to predict_year-1)
    yl = yield_df[yield_df["year"] <= (predict_year - 1)].copy()

    # intersect only on counties (not years)
    counties = set(yl["county"]) & set(nd["county"]) & set(wx["county"])

    return (
        yl[yl["county"].isin(counties)],
        nd[nd["county"].isin(counties)],
        wx[wx["county"].isin(counties)],
    )


def clean_storm_partitioned(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize storm data to a common schema used by feature builder:
      county (lowercase), datetime (timestamp), year (int), wind_mph (float)

    Supports both schemas:
      - Old (2013-2024): county, date, year, wind_mph, event_count, severe_gust_58
      - New (2025 live): county, datetime, wind_mph
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["county", "datetime", "year", "wind_mph"])

    out = df.copy()

    # --- county ---
    if "county" not in out.columns:
        # try common alternatives if any
        for c in ["County", "county_name", "NAME"]:
            if c in out.columns:
                out = out.rename(columns={c: "county"})
                break
    out["county"] = (
        out["county"].astype(str).str.lower().str.strip().str.replace(" county", "", regex=False)
    )

    # --- datetime/date ---
    if "datetime" not in out.columns:
        if "date" in out.columns:
            out["datetime"] = pd.to_datetime(out["date"], errors="coerce")
        else:
            # nothing usable
            out["datetime"] = pd.NaT
    else:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")

    # --- year ---
    if "year" not in out.columns:
        out["year"] = out["datetime"].dt.year
    else:
        # keep existing year if valid, otherwise derive
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out.loc[out["year"].isna(), "year"] = out["datetime"].dt.year

    # --- wind_mph ---
    if "wind_mph" not in out.columns:
        # try common alternatives if any
        for c in ["windSpeed", "wind_speed_mph", "max_wind_mph"]:
            if c in out.columns:
                out = out.rename(columns={c: "wind_mph"})
                break
    out["wind_mph"] = pd.to_numeric(out["wind_mph"], errors="coerce")

    # drop bad rows
    out = out.dropna(subset=["county", "datetime", "year", "wind_mph"])

    # Ensure one record per county-day-year with max wind (this matches your feature logic expectation)
    out["date"] = out["datetime"].dt.floor("D")
    out = (
        out.groupby(["county", "year", "date"], as_index=False)["wind_mph"].max()
        .rename(columns={"date": "datetime"})  # keep as timestamp at midnight
    )
    # ADD THIS LINE
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")

    return out[["county", "datetime", "year", "wind_mph"]]

# def clean_storm_partitioned(storm_raw: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
#     """Normalize storm/wind events written by ingestion_container (dataset=storm).

#     Expected columns (best-effort):
#       - county (or CZ_NAME)
#       - datetime (or BEGIN_DATE_TIME)
#       - wind_mph (or MAGNITUDE)
#       - year (optional)

#     Returns: county, datetime, year, wind_mph
#     """
#     df = storm_raw.copy()

#     if "datetime" not in df.columns:
#         if "BEGIN_DATE_TIME" in df.columns:
#             df["datetime"] = pd.to_datetime(df["BEGIN_DATE_TIME"], errors="coerce")
#         else:
#             df["datetime"] = pd.NaT
#     df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

#     if "county" not in df.columns:
#         if "CZ_NAME" in df.columns:
#             df["county"] = df["CZ_NAME"]
#         else:
#             df["county"] = ""

#     df["county"] = (
#         df["county"].astype(str).str.lower().str.strip().str.replace(" county","", regex=False)
#     )

#     if "wind_mph" not in df.columns:
#         if "MAGNITUDE" in df.columns:
#             df["wind_mph"] = df["MAGNITUDE"]
#         else:
#             df["wind_mph"] = np.nan
#     df["wind_mph"] = pd.to_numeric(df["wind_mph"], errors="coerce")

#     if "year" not in df.columns:
#         df["year"] = df["datetime"].dt.year
#     df["year"] = pd.to_numeric(df["year"], errors="coerce")

#     df = df.dropna(subset=["county","datetime","year","wind_mph"])

#     if strict:
#         df = df[(df["wind_mph"] >= 10) & (df["wind_mph"] <= 200)]
    
#     # Deduplicate to 1 row per county-date-year
#     agg_cols = [c for c in ["temperature", "rain_mm", "et_mm", "water_balance_mm", "dewpoint_temperature"] if c in df.columns]
#     df["date"] = df["datetime"].dt.date
#     if agg_cols:
#         df = df.groupby(["county", "date", "year"], as_index=False)[agg_cols].mean()
#     else:
#         df = df.drop_duplicates(subset=["county", "date", "year"])

#     return df[["county","datetime","year","wind_mph"]].drop_duplicates()
