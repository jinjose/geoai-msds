from __future__ import annotations

from datetime import datetime
import pandas as pd
import ee

from ..geo import counties_fc


ERA5_DAILY_VARS = {
    "temperature": "temperature_2m",
    "rainfall": "total_precipitation_sum",
    "evapotranspiration": "total_evaporation_sum",
    "soil_moisture": "volumetric_soil_water_layer_1",
}

ERA5_HOURLY_VARS = {
    "temperature": "temperature_2m",
    "rainfall": "total_precipitation",
    "evapotranspiration": "evaporation",
    "soil_moisture": "volumetric_soil_water_layer_1",
}


def _rows_from_fc(fc: ee.FeatureCollection):
    return [f["properties"] for f in fc.getInfo()["features"]]


def fetch_era5(window_start: datetime, window_end: datetime, granularity: str, counties: ee.FeatureCollection) -> pd.DataFrame:
    """
    ERA5 ingestion.

    Key fix: For granularity=yearly we aggregate into ONE annual image and run reduceRegions once.
    """
    print(f"[ERA5] START granularity={granularity} window={window_start.date()} → {window_end.date()}")
    granularity = granularity.lower()

    if granularity == "yearly":
        start = ee.Date(window_start.strftime("%Y-%m-%d"))
        end = ee.Date(window_end.strftime("%Y-%m-%d"))

        coll = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end)

        temp_mean = coll.select(ERA5_DAILY_VARS["temperature"]).mean().rename("temperature")
        soil_mean = coll.select(ERA5_DAILY_VARS["soil_moisture"]).mean().rename("soil_moisture")
        rain_sum = coll.select(ERA5_DAILY_VARS["rainfall"]).sum().rename("rainfall")
        evap_sum = coll.select(ERA5_DAILY_VARS["evapotranspiration"]).sum().rename("evapotranspiration")

        annual = temp_mean.addBands([rain_sum, evap_sum, soil_mean])

        fc = annual.reduceRegions(
            collection=counties,
            reducer=ee.Reducer.mean(),
            scale=10000,
            tileScale=4,
        )

        year = int(window_start.year)
        props = {"year": year, "month": 12, "date": f"{year}-12-31"}
        fc = fc.map(lambda f: f.set(props))

        rows = _rows_from_fc(fc)
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["county_fips"] = df.get("GEOID")
        df["county_name"] = df.get("NAME")

        for c in ["temperature", "rainfall", "evapotranspiration", "soil_moisture"]:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        print(f"[ERA5] Yearly rows: {len(df)}")
        return df[
            ["county_fips","county_name","date","evapotranspiration","month",
             "rainfall","soil_moisture","temperature","year"]
        ].copy()

    if granularity == "daily":
        start = ee.Date(window_start.strftime("%Y-%m-%d"))
        end = ee.Date(window_end.strftime("%Y-%m-%d"))
        coll = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end)
        varmap = ERA5_DAILY_VARS

        def per_image(img):
            sel = img.select(list(varmap.values()))
            fc = sel.reduceRegions(
                collection=counties,
                reducer=ee.Reducer.mean(),
                scale=10000,
                tileScale=4,
            )
            return fc.map(lambda f: f.set({
                "date": img.date().format("YYYY-MM-dd"),
                "year": window_start.year,
                "month": window_start.month,
                "county_fips": f.get("GEOID"),
                "county_name": f.get("NAME"),
                "temperature": f.get(varmap["temperature"]),
                "rainfall": f.get(varmap["rainfall"]),
                "evapotranspiration": f.get(varmap["evapotranspiration"]),
                "soil_moisture": f.get(varmap["soil_moisture"]),
            }))

        fc = coll.map(per_image).flatten()

    elif granularity == "hourly":
        start = ee.Date(window_start.isoformat().replace("+00:00", "Z"))
        end = ee.Date(window_end.isoformat().replace("+00:00", "Z"))
        coll = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start, end)
        varmap = ERA5_HOURLY_VARS

        def per_image(img):
            sel = img.select(list(varmap.values()))
            fc = sel.reduceRegions(
                collection=counties,
                reducer=ee.Reducer.mean(),
                scale=10000,
                tileScale=4,
            )
            return fc.map(lambda f: f.set({
                "date": img.date().format("YYYY-MM-dd"),
                "year": window_start.year,
                "month": window_start.month,
                "county_fips": f.get("GEOID"),
                "county_name": f.get("NAME"),
                "temperature": f.get(varmap["temperature"]),
                "rainfall": f.get(varmap["rainfall"]),
                "evapotranspiration": f.get(varmap["evapotranspiration"]),
                "soil_moisture": f.get(varmap["soil_moisture"]),
            }))

        fc = coll.map(per_image).flatten()

    else:
        raise ValueError("ERA5 supported granularities: daily/hourly/yearly")

    rows = _rows_from_fc(fc)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    for c in ["temperature", "rainfall", "evapotranspiration", "soil_moisture"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    print(f"[ERA5] Rows returned: {len(df)}")
    return df[
        ["county_fips","county_name","date","evapotranspiration","month",
         "rainfall","soil_moisture","temperature","year"]
    ].copy()


def ingest_era5(state_fips: str, county_fips: str, start: datetime, end: datetime, granularity: str) -> pd.DataFrame:
    print(f"[ERA5] START granularity={granularity} window={start.date()} → {end.date()}")
    county_filter = None if str(county_fips).upper() == "ALL" else county_fips
    counties = counties_fc(state_fips, county_filter)
    return fetch_era5(start, end, granularity, counties)
