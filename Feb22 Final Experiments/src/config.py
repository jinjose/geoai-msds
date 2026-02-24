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
    "aug15": (8, 15)
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
