# tests/test_cleaning_pipeline.py

import pandas as pd
import numpy as np
import pytest

from features.cleaning import (
    clean_yield,
    clean_ndvi,
    smooth_county_ndvi,
    clean_weather,
    clean_storm,
    enforce_intersection_lenient,
)


# ============================================================
# CLEAN YIELD
# ============================================================

def test_clean_yield_basic():
    raw = pd.DataFrame({
        "County": ["Polk", "Linn"],
        "Year": ["2020", "2021"],
        "Value": ["200", "210"]
    })

    result = clean_yield(raw)

    assert len(result) == 2
    assert result["county"].iloc[0] == "polk"
    assert result["yield_bu_acre"].iloc[0] == 200


def test_clean_yield_strict_range():
    raw = pd.DataFrame({
        "County": ["Polk"],
        "Year": [2020],
        "Value": ["50"]
    })

    result = clean_yield(raw, strict=True)

    assert result.empty


def test_clean_yield_cv_filter():
    raw = pd.DataFrame({
        "County": ["Polk"],
        "Year": [2020],
        "Value": ["200"],
        "CV (%)": ["30"]
    })

    result = clean_yield(raw, strict=True)

    assert result.empty


# ============================================================
# CLEAN NDVI
# ============================================================

def test_clean_ndvi_value_filter():
    raw = pd.DataFrame({
        "county_name": ["Polk", "Polk"],
        "date": ["2020-05-01", "2020-05-10"],
        "NDVI": [-0.5, 0.8]
    })

    result = clean_ndvi(raw, strict=False)

    assert len(result) == 1
    assert result["NDVI"].iloc[0] == 0.8


def test_clean_ndvi_min_points():
    raw = pd.DataFrame({
        "county_name": ["Polk"] * 3,
        "date": pd.date_range("2020-05-01", periods=3),
        "NDVI": [0.5, 0.6, 0.7]
    })

    result = clean_ndvi(raw, min_points_per_county_year=6, strict=True)

    assert result.empty


# ============================================================
# NDVI SMOOTHING
# ============================================================

def test_smooth_county_ndvi():
    raw = pd.DataFrame({
        "county": ["polk"] * 10,
        "year": [2020] * 10,
        "date": pd.date_range("2020-05-01", periods=10),
        "NDVI": np.linspace(0.4, 0.8, 10)
    })

    result = smooth_county_ndvi(raw, window=5)

    assert "NDVI_smooth" in result.columns
    assert len(result) == 10


# ============================================================
# CLEAN WEATHER
# ============================================================

def test_clean_weather_kelvin_and_rain_conversion():
    raw = pd.DataFrame({
        "county_name": ["Polk"],
        "date": ["2020-06-01"],
        "temperature": [300],  # Kelvin
        "rainfall": [0.01],    # meters
        "evapotranspiration": [0.005]
    })

    result = clean_weather(raw, strict=False)

    assert result["temperature"].iloc[0] < 100  # converted to Celsius
    assert result["rain_mm"].iloc[0] == 10.0
    assert "water_balance_mm" in result.columns


def test_clean_weather_growing_season_filter():
    raw = pd.DataFrame({
        "county_name": ["Polk", "Polk"],
        "date": ["2020-01-01", "2020-06-01"],
        "temperature": [20, 25],
        "rainfall": [0.01, 0.02],
        "evapotranspiration": [0.005, 0.005]
    })

    result = clean_weather(raw, strict=False)

    assert len(result) == 1
    assert result["date"].dt.month.iloc[0] == 6


# ============================================================
# CLEAN STORM
# ============================================================

def test_clean_storm_event_filter():
    raw = pd.DataFrame({
        "CZ_NAME": ["Polk", "Polk"],
        "BEGIN_DATE_TIME": ["2020-06-01 10:00", "2020-06-01 11:00"],
        "MAGNITUDE": [50, 60],
        "EVENT_TYPE": ["Thunderstorm Wind", "Tornado"]
    })

    result = clean_storm(raw)

    assert len(result) == 1
    assert result["wind_mph"].iloc[0] == 50


# ============================================================
# ENFORCE INTERSECTION
# ============================================================

def test_enforce_intersection_lenient():
    yield_df = pd.DataFrame({
        "county": ["polk", "linn"],
        "year": [2020, 2020],
        "yield_bu_acre": [200, 210]
    })

    ndvi_df = pd.DataFrame({
        "county": ["polk"],
        "year": [2020],
        "NDVI": [0.6]
    })

    wx_df = pd.DataFrame({
        "county": ["polk"],
        "year": [2020],
        "temperature": [25]
    })

    y, n, w = enforce_intersection_lenient(yield_df, ndvi_df, wx_df)

    assert len(y) == 1
    assert y["county"].iloc[0] == "polk"