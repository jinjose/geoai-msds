from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
import ee

from ..geo import counties_fc


def latest_image_upto_year(collection_id: str, year: int) -> ee.Image:
    """Return the latest image in an ImageCollection with image-year <= `year`."""
    coll = (
        ee.ImageCollection(collection_id)
        .filter(ee.Filter.calendarRange(2000, year, "year"))
        .sort("system:time_start", False)
    )
    if coll.size().getInfo() == 0:
        raise RuntimeError(f"No images found in {collection_id} up to year={year}")
    return ee.Image(coll.first())


def corn_mask_for_year(year: int) -> ee.Image:
    """CDL often lags. Fall back to the latest available CDL year <= requested year."""
    img = latest_image_upto_year("USDA/NASS/CDL", year)
    used_year = ee.Date(img.get("system:time_start")).get("year").getInfo()
    print(f"[NDVI] Using CDL year={used_year} for requested year={year}")
    cdl = img.select("cropland")
    return cdl.eq(1)  # corn


def modis_ndvi_collection_upto_year(year: int) -> ee.ImageCollection:
    return (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filter(ee.Filter.calendarRange(2000, year, "year"))
    )


def fetch_ndvi_daily_like(
    window_start: datetime,
    window_end: datetime,
    counties: ee.FeatureCollection,
) -> pd.DataFrame:
    """Return one row per (date, county) within the window."""
    year = window_start.year
    corn_mask = corn_mask_for_year(year)

    coll = (
        modis_ndvi_collection_upto_year(year)
        .filterDate(window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"))
        .select("NDVI")
    )

    def per_image(image):
        masked = image.updateMask(corn_mask)

        def per_county(county):
            stats = masked.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=county.geometry(),
                scale=250,
                bestEffort=True,
                maxPixels=1e13,
            )
            return ee.Feature(
                None,
                {
                    "date": image.date().format("YYYY-MM-dd"),
                    "year": year,
                    "county_name": county.get("NAME"),
                    "geoid": county.get("GEOID"),
                    "NDVI": stats.get("NDVI"),
                },
            )

        return counties.map(per_county)

    fc = coll.map(per_image).flatten()

    # Your pipeline uses NDVI_GRANULARITY=yearly, which aggregates and stays small.
    rows = [f["properties"] for f in fc.getInfo()["features"]]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["year"] = df["year"].astype(int)
    df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce") / 10000.0

    return df[["NDVI", "county_name", "date", "geoid", "year"]]


def fetch_ndvi(
    window_start: datetime,
    window_end: datetime,
    granularity: str,
    counties: ee.FeatureCollection,
) -> pd.DataFrame:
    print(f"[NDVI] START granularity={granularity} window={window_start.date()} → {window_end.date()}")
    granularity = granularity.lower()

    if granularity == "daily":
        return fetch_ndvi_daily_like(window_start, window_end, counties)

    if granularity == "hourly":
        day_start = window_start.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        df = fetch_ndvi_daily_like(day_start, day_end, counties)
        if df.empty:
            return df
        df["hour"] = int(window_start.hour)
        print(f"[NDVI] Hourly rows: {len(df)}")
        return df[["NDVI", "county_name", "date", "geoid", "year", "hour"]]

    if granularity == "yearly":
        df = fetch_ndvi_daily_like(window_start, window_end, counties)
        if df.empty:
            return df

        out = (
            df.groupby(["geoid", "county_name", "year"], as_index=False)
            .agg(NDVI=("NDVI", "mean"))
        )
        print(f"[NDVI] Yearly rows: {len(out)}")
        out["date"] = out["year"].astype(str) + "-12-31"
        return out[["NDVI", "county_name", "date", "geoid", "year"]]

    raise ValueError("NDVI supported granularities: daily/hourly/yearly")


def ingest_ndvi(
    state_fips: str,
    county_fips: str,
    start: datetime,
    end: datetime,
    granularity: str,
) -> pd.DataFrame:
    county_filter = None if str(county_fips).upper() == "ALL" else county_fips
    counties = counties_fc(state_fips, county_filter)
    return fetch_ndvi(start, end, granularity, counties)
