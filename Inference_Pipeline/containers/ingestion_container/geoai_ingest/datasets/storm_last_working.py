import os
from datetime import datetime

# ----------------------------
# Incremental ingestion helpers (DynamoDB)
# ----------------------------
import boto3
from datetime import timedelta

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
import requests
import pandas as pd
import shapefile
from shapely.geometry import shape as _shape, Point

from ..s3io import parse_s3, write_csv_parquet, partition_prefix, S3Base

STORM_ARCGIS_URL = os.environ.get(
    "STORM_ARCGIS_URL",
    "https://services.arcgis.com/jIL9msH9OI208GCb/arcgis/rest/services/NOAA_Storm_Events_Database_1950-2021_v2/FeatureServer/0/query"
)

def arcgis_post(url: str, data: dict, timeout: int = 60) -> dict:
    r = requests.post(url, data=data, timeout=timeout)
    r.raise_for_status()
    return r.json()

def storm_object_ids(start: str, end: str) -> list[int]:
    payload = {
        "where": f"BEGIN_DATE >= DATE '{start}' AND BEGIN_DATE <= DATE '{end}'",
        "returnIdsOnly": "true",
        "f": "json",
    }
    out = arcgis_post(STORM_ARCGIS_URL, payload)
    return out.get("objectIds", []) or []

def storm_fetch_by_ids(ids: list[int]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()

    payload = {
        "objectIds": ",".join(map(str, ids)),
        "outFields": "*",
        "returnGeometry": "true",
        "f": "json",
    }
    out = arcgis_post(STORM_ARCGIS_URL, payload)
    feats = out.get("features", []) or []
    rows = []
    for f in feats:
        props = f.get("attributes", {}) or {}
        geom = f.get("geometry", {}) or {}
        props["geom_x"] = geom.get("x")
        props["geom_y"] = geom.get("y")
        rows.append(props)
    return pd.DataFrame(rows)

def _ensure_county_shapefile_available(region: str) -> str:
    """
    Returns directory containing tl_2023_us_county.{shp,shx,dbf}
    Priority:
      1) bundled inside image at /opt/program/data/counties
      2) COUNTY_SHP_DIR env (if you mount something else)
      3) Download from COUNTY_SHP_S3_URI if files missing (prefix with those 3 files)
    """
    shp_dir = os.environ.get("COUNTY_SHP_DIR", "/opt/program/data/counties")
    os.makedirs(shp_dir, exist_ok=True)

    need = ["tl_2023_us_county.shp", "tl_2023_us_county.shx", "tl_2023_us_county.dbf"]
    missing = [f for f in need if not os.path.exists(os.path.join(shp_dir, f))]
    if not missing:
        return shp_dir

    s3_uri = os.environ.get("COUNTY_SHP_S3_URI")
    if not s3_uri:
        raise FileNotFoundError(f"Missing shapefile parts: {missing} in {shp_dir} and COUNTY_SHP_S3_URI not set")

    base = parse_s3(s3_uri)
    import boto3
    s3 = boto3.client("s3", region_name=region)

    for f in need:
        key = base.prefix + f
        local = os.path.join(shp_dir, f)
        s3.download_file(base.bucket, key, local)

    return shp_dir

def _load_county_polygons(region: str):
    shp_dir = _ensure_county_shapefile_available(region)
    shp_path = os.path.join(shp_dir, "tl_2023_us_county.shp")

    r = shapefile.Reader(shp_path)
    fields = [f[0] for f in r.fields[1:]]
    polys = []
    for sr in r.shapeRecords():
        rec = dict(zip(fields, sr.record))
        geom = _shape(sr.shape.__geo_interface__)
        polys.append((rec, geom))
    return polys

def _point_to_county(polys, lon: float, lat: float):
    pt = Point(lon, lat)
    for rec, geom in polys:
        if geom.contains(pt):
            return rec
    return None

def ingest_storm_daily(out: S3Base, region: str, state_fips: str, county_fips: str,
                       start: datetime | None, end: datetime | None) -> None:
    """
    Storm ingestion (daily).

    If start/end are None and REGISTRY_TABLE is set, this will do incremental ingestion:
      start = last_ingested_date + 1 day (or BACKFILL_START_YEAR-01-01 if none)
      end   = today (UTC)

    Registry is updated only after a successful write with non-empty data.
    """
    dataset = "storm"

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
        print(f"[STORM] incremental window {start.date()} → {end.date()} (last={last})")

    print(f"[STORM] START state={state_fips} county={county_fips} window={start.date()} → {end.date()}")
    polys = _load_county_polygons(region)
    print(f"[STORM] Loaded county polygons")

    ids = storm_object_ids(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(f"[STORM] objectIds fetched: {len(ids)}")
    df = storm_fetch_by_ids(ids)
    print(f"[STORM] Fetched storm data: {len(df)} rows")
    if df.empty:
        print(f"[STORM] No storm data found for the specified window.")
        return

    matches = []
    for _, row in df.iterrows():
        x, y = row.get("geom_x"), row.get("geom_y")
        if pd.isna(x) or pd.isna(y):
            matches.append((None, None))
            continue
        rec = _point_to_county(polys, float(x), float(y))
        if rec is None:
            matches.append((None, None))
        else:
            matches.append((rec.get("STATEFP"), rec.get("COUNTYFP")))

    df["STATEFP"] = [a for a, _ in matches]
    df["COUNTYFP"] = [b for _, b in matches]

    df = df[df["STATEFP"] == str(state_fips).zfill(2)]
    if county_fips.upper() != "ALL":
        df = df[df["COUNTYFP"] == str(county_fips).zfill(3)]
        print(f"[STORM] Rows after state/county filter: {len(df)}")

    if df.empty:
        print("[STORM] No rows after filtering — skipping write")
        return

    key_prefix = partition_prefix(out, "storm", str(state_fips).zfill(2), county_fips, "daily",
                                  year=start.year, month=start.month, day=start.day)
    print(f"[STORM] Writing to: s3://{out.bucket}/{key_prefix}")

    write_csv_parquet(region, out, key_prefix, df, basename="part")
    print(f"[STORM] DONE — wrote {len(df)} rows")

    _registry_put_date(dataset, end.strftime("%Y-%m-%d"))
    print(f"[STORM] registry updated last_ingested_date={end.strftime('%Y-%m-%d')}")
