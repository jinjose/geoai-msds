import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================
def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "src").exists():
            return p
    raise RuntimeError("Project root not found")

PROJECT_ROOT = find_project_root(Path(__file__))
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURE_DIR = PROJECT_ROOT / "data" / "features_frozen"
FEATURE_DIR.mkdir(exist_ok=True)

# ============================================================
# IMPORTS
# ============================================================
from config import FORECAST_CUTOFFS
from preprocessing.cleaning import (
    clean_yield,
    clean_ndvi,
    clean_weather,
    smooth_county_ndvi,
)
from features.build_features import build_feature_table


# ============================================================
# COUNTY NORMALIZATION
# ============================================================
def normalize_county(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(" county", "", regex=False)
    )


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\nLoading raw datasets...")

    yield_raw = pd.read_csv(DATA_DIR / "IOWA_county_wise_yield.csv") #added 2008 last 5 year trend
    ndvi_raw = pd.read_csv(DATA_DIR / "corn_ndvi_iowa_county_all_years.csv", parse_dates=["date"])
    wx_raw = pd.read_csv(DATA_DIR / "era5_iowa_county_all_years.csv", parse_dates=["date"])

    print("\nCleaning datasets...")

    yield_df = clean_yield(yield_raw)        # FULL history preserved
    ndvi = clean_ndvi(ndvi_raw)
    wx = clean_weather(wx_raw)

    # Normalize counties
    yield_df["county"] = normalize_county(yield_df["county"])
    ndvi["county"] = normalize_county(ndvi["county"])
    wx["county"] = normalize_county(wx["county"])

    # Smooth NDVI
    ndvi = smooth_county_ndvi(ndvi, window=9, poly=2)

    # ========================================================
    # ALIGN ONLY NDVI + WEATHER
    # (Do NOT intersect yield here)
    # ========================================================
    common_counties = (
        set(ndvi["county"])
        .intersection(set(wx["county"]))
    )

    common_years = (
        set(ndvi["year"])
        .intersection(set(wx["year"]))
    )

    ndvi = ndvi[
        (ndvi["county"].isin(common_counties)) &
        (ndvi["year"].isin(common_years))
    ]

    wx = wx[
        (wx["county"].isin(common_counties)) &
        (wx["year"].isin(common_years))
    ]

    print("\nAfter NDVI/WX alignment:")
    print("Counties:", len(common_counties))
    print("Years:", sorted(common_years))

    # Historical baselines
    ndvi_hist_mean = ndvi.groupby("county")["NDVI"].mean()
    temp_hist_mean = wx.groupby("county")["temperature"].mean()

    print("\nStarting feature generation...")

    # ========================================================
    # BUILD + SAVE FEATURES
    # ========================================================
    for cutoff_name, (month, day) in FORECAST_CUTOFFS.items():

        print(f"\nBuilding features for {cutoff_name}")

        feature_df = build_feature_table(
            yield_df,     # FULL yield history passed
            ndvi,
            wx,
            month,
            day,
            ndvi_hist_mean,
            temp_hist_mean,
        )

        if feature_df.empty:
            print("No features generated.")
            continue

        print("Feature shape:", feature_df.shape)

        out_path = FEATURE_DIR / f"features_{cutoff_name}.csv"
        feature_df.to_csv(out_path, index=False)

        print(f"Saved to {out_path}")

    print("\nFeature generation complete.")
