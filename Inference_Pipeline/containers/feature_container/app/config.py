"""
Global configuration for Iowa corn yield capstone
"""

# =========================================
# FORECAST CUTOFFS
# =========================================
# Forecast date relative to growing season.
# Models only see data up to this date.
#
# Format:
#   cutoff_name: (month, day)
#
FORECAST_CUTOFFS = {
    "jun01": (6, 1),
    "jul01": (7, 1),
    "jul15": (7, 15),
    "aug01": (8, 1),
    "aug15": (8, 15),
}

# =========================================
# TARGET COLUMN
# =========================================
TARGET_COL = "yield_bu_acre"

# =========================================
# IDENTIFIER COLUMNS
# =========================================
ID_COLS = [
    "county",
    "year",
]

# =========================================
# RANDOM STATE (REPRODUCIBILITY)
# =========================================
RANDOM_STATE = 42

# =========================================
# NDVI SMOOTHING DEFAULTS
# =========================================
NDVI_SMOOTHING = {
    "method": "savgol",
    "freq": "7D",
    "window": 9,
    "poly": 2,
    "max_gap_points": 2,
}

# =========================================
# WEATHER FILTERING
# =========================================
GROWING_MONTHS = (4, 9)
MIN_WEATHER_DAYS = 150

# =========================================
# NDVI QUALITY CONTROL
# =========================================
MIN_NDVI_POINTS_PER_YEAR = 6

# =========================================
# YIELD CLEANING
# =========================================
MAX_YIELD_CV_PCT = 25



# =========================================
# FEATURE LISTS (INFERENCE)
# Stored as CSVs under app/feature_lists/features_<season>.csv
# =========================================
FEATURE_LIST_DIR = "app/feature_lists"


# =========================================
# FEATURE LIST LOADER (per season)
# =========================================

import json
from pathlib import Path

def load_expected_features_from_schema(feature_season: str, schema_path: str | None = None) -> list[str]:
    """
    Permanent source of truth for inference features:
    - Uses model feature schema JSON (expected_features)
    - Filters out target if present
    """
    season = str(feature_season).lower().strip()

    # Default: keep schemas versioned in-repo (good for local).
    # For AWS inference, pass schema_path as an S3-downloaded local path.
    if schema_path is None:
        schema_path = str(Path(__file__).resolve().parent / season / "feature_schema.json")

    p = Path(schema_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing feature schema json at: {p}")

    with p.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    # Optional: enforce season match if schema has it
    cutoff = str(schema.get("cutoff", "")).lower().strip()
    if cutoff and cutoff != season:
        raise ValueError(f"Schema cutoff={cutoff} does not match requested season={season}")

    expected = schema.get("expected_features", [])
    if not expected:
        raise ValueError(f"Schema has no expected_features: {p}")

    target = str(schema.get("target", "yield_bu_acre"))

    # expected_features already includes county per your schema
    feats = [c for c in expected if c and c != target and c.lower() not in ("yield", "target")]

    return feats

# import pandas as _pd
# from pathlib import Path as _Path

# def load_feature_list(feature_season: str) -> list[str]:
#     season = str(feature_season).lower().strip()
#     p = _Path(__file__).resolve().parent / "feature_lists" / f"features_{season}.csv"
#     if not p.exists():
#         raise FileNotFoundError(f"Missing feature list for season={season}: {p}")
#     df = _pd.read_csv(p)
#     if df.shape[1] == 1:
#         feats = df.iloc[:, 0].astype(str).tolist()
#     elif "feature" in df.columns:
#         feats = df["feature"].astype(str).tolist()
#     else:
#         feats = df.iloc[:, 0].astype(str).tolist()
#     feats = [f for f in feats if f and f.lower() not in ("yield_bu_acre", "yield", "target")]
#     return feats
