import pandas as pd
import numpy as np
import pytest

import features.feature_table_logic as ft
from features.feature_table_logic import feature_table_logic


# ============================================================
# PATCH cutoff_mask TO ALWAYS KEEP ROWS
# ============================================================

@pytest.fixture(autouse=True)
def patch_cutoff_mask(monkeypatch):
    monkeypatch.setattr(
        ft,
        "cutoff_mask",
        lambda dates, m, d: np.ones(len(dates), dtype=bool)
    )


# ============================================================
# HISTORICAL MODE TEST
# ============================================================

def test_feature_table_historical_basic():

    yield_df = pd.DataFrame({
        "county": ["polk", "polk"],
        "year": [2019, 2020],
        "yield_bu_acre": [180, 200]
    })

    ndvi = pd.DataFrame({
        "county": ["polk"] * 5,
        "year": [2020] * 5,
        "date": pd.date_range("2020-05-01", periods=5),
        "NDVI": [0.5, 0.6, 0.7, 0.8, 0.75]
    })

    wx = pd.DataFrame({
        "county": ["polk"] * 10,
        "year": [2020] * 10,
        "date": pd.date_range("2020-04-01", periods=10),
        "temperature": [30] * 10,
        "rain_mm": [5] * 10,
        "water_balance_mm": [2] * 10
    })

    storm_df = pd.DataFrame({
        "county": ["polk"],
        "year": [2020],
        "datetime": pd.to_datetime(["2020-06-01"]),
        "wind_mph": [60]
    })

    temp_hist_mean = pd.Series({"polk": 25})

    result = feature_table_logic(
        yield_df,
        ndvi,
        wx,
        storm_df,
        cutoff_month=7,
        cutoff_day=1,
        temp_hist_mean=temp_hist_mean,
        mode="historical"
    )

    assert not result.empty
    assert "yield_bu_acre" in result.columns
    assert "lag1_yield" in result.columns
    assert result["ndvi_peak"].iloc[0] == 0.8
    assert result["heat_days_gt32"].iloc[0] == 10
    assert result["wind_severe_days_58_cutoff"].iloc[0] == 1


# ============================================================
# LIVE MODE TEST
# ============================================================

def test_feature_table_live_mode():

    yield_df = pd.DataFrame({
        "county": ["polk", "polk"],
        "year": [2019, 2020],
        "yield_bu_acre": [180, 200]
    })

    ndvi = pd.DataFrame({
        "county": ["polk"] * 5,
        "year": [2020] * 5,
        "date": pd.date_range("2020-05-01", periods=5),
        "NDVI": [0.5, 0.6, 0.7, 0.8, 0.75]
    })

    wx = pd.DataFrame({
        "county": ["polk"] * 5,
        "year": [2020] * 5,
        "date": pd.date_range("2020-04-01", periods=5),
        "temperature": [30] * 5,
        "rain_mm": [5] * 5,
        "water_balance_mm": [2] * 5
    })

    # Proper datetime dtype even if empty
    storm_df = pd.DataFrame({
        "county": [],
        "year": [],
        "datetime": pd.to_datetime([]),
        "wind_mph": []
    })

    temp_hist_mean = pd.Series({"polk": 25})

    result = feature_table_logic(
        yield_df,
        ndvi,
        wx,
        storm_df,
        cutoff_month=7,
        cutoff_day=1,
        temp_hist_mean=temp_hist_mean,
        mode="live"
    )

    assert not result.empty
    assert "yield_bu_acre" not in result.columns
    assert "lag1_yield" not in result.columns
    assert "rolling_3yr_mean" in result.columns


# ============================================================
# NDVI SLOPE TEST
# ============================================================

def test_ndvi_slope_computation():

    yield_df = pd.DataFrame({
        "county": ["polk"],
        "year": [2020],
        "yield_bu_acre": [200]
    })

    ndvi = pd.DataFrame({
        "county": ["polk"] * 2,
        "year": [2020] * 2,
        "date": pd.to_datetime(["2020-05-01", "2020-06-01"]),
        "NDVI": [0.5, 0.7]
    })

    wx = pd.DataFrame({
        "county": ["polk"] * 5,
        "year": [2020] * 5,
        "date": pd.date_range("2020-04-01", periods=5),
        "temperature": [25] * 5,
        "rain_mm": [5] * 5,
        "water_balance_mm": [1] * 5
    })

    # Proper datetime dtype even if empty
    storm_df = pd.DataFrame({
        "county": [],
        "year": [],
        "datetime": pd.to_datetime([]),
        "wind_mph": []
    })

    temp_hist_mean = pd.Series({"polk": 20})

    result = feature_table_logic(
        yield_df,
        ndvi,
        wx,
        storm_df,
        cutoff_month=7,
        cutoff_day=1,
        temp_hist_mean=temp_hist_mean,
        mode="historical"
    )

    assert not result.empty
    assert not np.isnan(result["ndvi_slope"].iloc[0])
    assert result["ndvi_peak"].iloc[0] == 0.7