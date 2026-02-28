import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_yield_raw():
    return pd.DataFrame({
        "County": ["Polk", "Linn"],
        "Year": ["2020", "2021"],
        "Value": ["200", "210"]
    })


@pytest.fixture
def sample_ndvi_raw():
    return pd.DataFrame({
        "county_name": ["Polk"] * 8,
        "date": pd.date_range("2020-05-01", periods=8),
        "NDVI": np.linspace(0.4, 0.8, 8)
    })


@pytest.fixture
def sample_weather_raw():
    return pd.DataFrame({
        "county_name": ["Polk"] * 200,
        "date": pd.date_range("2020-04-01", periods=200),
        "temperature": [300] * 200,
        "rainfall": [0.01] * 200,
        "evapotranspiration": [0.005] * 200
    })


@pytest.fixture
def sample_storm_raw():
    return pd.DataFrame({
        "CZ_NAME": ["Polk"],
        "BEGIN_DATE_TIME": ["2020-06-01 10:00"],
        "MAGNITUDE": [50],
        "EVENT_TYPE": ["Thunderstorm Wind"]
    })