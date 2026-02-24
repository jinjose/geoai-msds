import argparse
import pandas as pd
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
# ARGUMENT PARSING
# ============================================================
parser = argparse.ArgumentParser(description="Build county-level features")

parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["historical", "live"],
)

args = parser.parse_args()
mode = args.mode.lower()


# ============================================================
# PATH SETUP
# ============================================================
PROJECT_ROOT = find_project_root(Path(__file__))

if mode == "historical":
    DATA_DIR = PROJECT_ROOT / "training-dataset" / "raw"
    FEATURE_DIR = PROJECT_ROOT / "training-dataset" / "features_frozen"
else:
    DATA_DIR = PROJECT_ROOT / "inference-dataset" / "raw"
    FEATURE_DIR = PROJECT_ROOT / "inference-dataset" / "features_frozen"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nMode: {mode}")
print(f"Reading from: {DATA_DIR}")
print(f"Saving to: {FEATURE_DIR}")


# ============================================================
# IMPORTS
# ============================================================
from config import FORECAST_CUTOFFS
from features.cleaning import (
    clean_yield,
    clean_ndvi,
    clean_weather,
    smooth_county_ndvi,
    clean_storm,
)
from features.feature_table_logic import feature_table_logic


# ============================================================
# LOAD RAW DATA
# ============================================================
yield_raw = pd.read_csv(DATA_DIR / "IOWA_county_wise_yield.csv")

ndvi_raw = pd.read_csv(
    DATA_DIR / "corn_ndvi_iowa_state.csv",
    parse_dates=["date"],
)

wx_raw = pd.read_csv(
    DATA_DIR / "era5_iowa_state.csv",
    parse_dates=["date"],
)

storm_raw = pd.read_csv(DATA_DIR / "iowa_storm_events.csv")


# ============================================================
# CLEAN DATA
# ============================================================
yield_df = clean_yield(yield_raw)
ndvi = clean_ndvi(ndvi_raw)
wx = clean_weather(wx_raw)
storm_df = clean_storm(storm_raw, strict=True)

# Normalize counties everywhere (extra safety)
yield_df["county"] = normalize_county(yield_df["county"])
ndvi["county"] = normalize_county(ndvi["county"])
wx["county"] = normalize_county(wx["county"])
storm_df["county"] = normalize_county(storm_df["county"])


# ============================================================
# HISTORICAL MODE: FULL PANEL GRID
# ============================================================
if mode == "historical":

    print("Historical mode: preserving full yield history and repairing gaps.")

    min_year = int(yield_df["year"].min())
    max_year = int(yield_df["year"].max())

    all_counties = yield_df["county"].unique()
    all_years = range(min_year, max_year + 1)

    multi_index = pd.MultiIndex.from_product(
        [all_counties, all_years],
        names=["county", "year"],
    )

    yield_df = (
        yield_df
        .set_index(["county", "year"])
        .reindex(multi_index)
        .reset_index()
    )

    # Iowa state proxy
    iowa_state_yields = {
        2008: 171.0, 2009: 165.0, 2010: 165.0, 2011: 172.0,
        2012: 137.0, 2013: 164.0, 2014: 178.0, 2015: 192.0,
        2016: 203.0, 2017: 202.0, 2018: 196.0, 2019: 198.0,
        2020: 177.0, 2021: 204.0, 2022: 200.0, 2023: 201.0,
        2024: 211.0,
    }

    state_series = pd.Series(iowa_state_yields)

    missing_mask = yield_df["yield_bu_acre"].isna()
    yield_df.loc[missing_mask, "yield_bu_acre"] = (
        yield_df.loc[missing_mask, "year"].map(state_series)
    )

else:
    print("Live mode: no grid forcing and no statewide imputation.")


# ============================================================
# SMOOTH NDVI
# ============================================================
ndvi = smooth_county_ndvi(ndvi, window=9, poly=2)


# ============================================================
# ALIGN NDVI + WEATHER
# ============================================================
common_counties = set(ndvi["county"]).intersection(set(wx["county"]))
common_years = set(ndvi["year"]).intersection(set(wx["year"]))

ndvi = ndvi[
    (ndvi["county"].isin(common_counties)) &
    (ndvi["year"].isin(common_years))
]

wx = wx[
    (wx["county"].isin(common_counties)) &
    (wx["year"].isin(common_years))
]


# ============================================================
# TEMPERATURE BASELINE
# ============================================================
temp_hist_mean = wx.groupby("county")["temperature"].mean()


# ============================================================
# BUILD FEATURES
# ============================================================
print("\nStarting feature generation...")

for cutoff_name, (month, day) in FORECAST_CUTOFFS.items():

    print(f"\nBuilding features for {cutoff_name}")

    feature_df = feature_table_logic(
        yield_df=yield_df,
        ndvi=ndvi,
        wx=wx,
        storm_df=storm_df,
        cutoff_month=month,
        cutoff_day=day,
        temp_hist_mean=temp_hist_mean,
        mode=mode,
    )

    if feature_df.empty:
        print("No features generated.")
        continue

    out_path = FEATURE_DIR / f"features_{cutoff_name}.csv"
    feature_df.to_csv(out_path, index=False)

    print(f"Saved to {out_path}")

print("\nFeature generation complete.")