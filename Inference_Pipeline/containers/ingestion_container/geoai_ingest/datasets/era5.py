from __future__ import annotations

from datetime import datetime
import pandas as pd
import ee

from ..geo import counties_fc


# ----------------------------
# Incremental ingestion helpers (DynamoDB)
# ----------------------------
import os
from datetime import timedelta
import boto3

def _aws_region() -> str:
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("AWS_REGION_NAME")
        or "ap-south-1"   # pick your default
    )

def _dd_table():
    name = os.environ["REGISTRY_TABLE"]
    session = boto3.session.Session(region_name=_aws_region())
    return session.resource("dynamodb").Table(name)

def _registry_get_date(dataset: str):
    tbl = _dd_table()
    if tbl is None:
        return None
    resp = tbl.get_item(Key={"pk": dataset})
    return (resp.get("Item") or {}).get("last_ingested_date")

def _registry_put_date(dataset: str, last_date_str: str):
    tbl = _dd_table()
    if tbl is None:
        return
    tbl.put_item(Item={"pk": dataset, "last_ingested_date": last_date_str})



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


import ee

def _rows_from_fc(fc: ee.FeatureCollection, page_size: int = 4500):
    # page_size < 5000 to be safe
    total = int(fc.size().getInfo())
    rows = []
    for offset in range(0, total, page_size):
        chunk = ee.FeatureCollection(fc.toList(page_size, offset))
        info = chunk.getInfo()
        rows.extend([f["properties"] for f in info.get("features", [])])
    return rows


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
        # NOTE:
        # Avoid building a single huge FeatureCollection via coll.map(...).flatten(),
        # which can exceed Earth Engine's 5000-element limit for longer windows.
        # Instead, iterate image-by-image (daily) and collect rows safely.

        varmap = ERA5_DAILY_VARS
        coll = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(
            ee.Date(window_start.strftime("%Y-%m-%d")),
            ee.Date(window_end.strftime("%Y-%m-%d")),
        )

        images = coll.toList(coll.size())
        n = images.size().getInfo()
        rows: list[dict] = []

        for i in range(n):
            img = ee.Image(images.get(i))
            sel = img.select(list(varmap.values()))

            fc = sel.reduceRegions(
                collection=counties,
                reducer=ee.Reducer.mean(),
                scale=10000,
                tileScale=4,
            )

            fc = fc.map(lambda f: f.set({
                "date": img.date().format("YYYY-MM-dd"),
                # derive correct year/month from image date (window may span months)
                "year": ee.Date(img.date()).get("year"),
                "month": ee.Date(img.date()).get("month"),
                "county_fips": f.get("GEOID"),
                "county_name": f.get("NAME"),
                "temperature": f.get(varmap["temperature"]),
                "rainfall": f.get(varmap["rainfall"]),
                "evapotranspiration": f.get(varmap["evapotranspiration"]),
                "soil_moisture": f.get(varmap["soil_moisture"]),
            }))

            rows.extend(_rows_from_fc(fc))

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


def ingest_era5(state_fips: str, county_fips: str, start: datetime | None, end: datetime | None, granularity: str) -> pd.DataFrame:
    """
    Ingest ERA5 for a window.

    If start/end are None and REGISTRY_TABLE is set, this will do incremental ingestion:
      start = last_ingested_date + 1 day (or BACKFILL_START_YEAR-01-01 if none)
      end   = today (UTC)

    Note: for granularity='yearly', this will ingest one annual aggregate for the given window year.
    """
    dataset = "era5"

    # Incremental window if not provided
    if start is None or end is None:
        last = _registry_get_date(dataset)
        if last:
            start_date = datetime.strptime(last, "%Y-%m-%d").date() + timedelta(days=1)
        else:
            backfill_year = int(os.environ.get("BACKFILL_START_YEAR", "2014"))
            start_date = datetime(backfill_year, 1, 1).date()

        end_date = datetime.utcnow().date()
        start = datetime(start_date.year, start_date.month, start_date.day)
        end = datetime(end_date.year, end_date.month, end_date.day)        
        print(f"[ERA5] incremental window {start.date()} → {end.date()} (last={last})")
        if start.date() > end.date():
            print(f"[NDVI] nothing new to ingest (start={start.date()} > end={end.date()}); skipping")
            return pd.DataFrame()

    print(f"[ERA5] START granularity={granularity} window={start.date()} → {end.date()}")
    county_filter = None if str(county_fips).upper() == "ALL" else county_fips
    counties = counties_fc(state_fips, county_filter)
    df = fetch_era5(start, end, granularity, counties)

    # Update registry only if we actually produced rows
    if df is not None and not df.empty:
        _registry_put_date(dataset, end.strftime("%Y-%m-%d"))
        print(f"[ERA5] registry updated last_ingested_date={end.strftime('%Y-%m-%d')}")
    else:
        print("[ERA5] no rows produced; registry not updated")

    return df
