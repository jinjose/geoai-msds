import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple, Set

import boto3
import pandas as pd
import requests

from ..s3io import parse_s3, write_csv_parquet, partition_prefix, S3Base

# -------------------------------------------------------------------
# DynamoDB registry helpers (date-only schema)
# Table schema: pk (String), last_ingested_date (YYYY-MM-DD)
# -------------------------------------------------------------------

def _aws_region() -> str:
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("AWS_REGION_NAME")
        or "ap-south-1"
    )

def _dd_table():
    """
    Returns DynamoDB table object if REGISTRY_TABLE env is set; otherwise None.
    """
    name = os.environ.get("REGISTRY_TABLE")
    if not name:
        return None
    session = boto3.session.Session(region_name=_aws_region())
    return session.resource("dynamodb").Table(name)

def _registry_get_date(dataset: str) -> Optional[str]:
    tbl = _dd_table()
    if tbl is None:
        return None
    resp = tbl.get_item(Key={"pk": dataset})
    return (resp.get("Item") or {}).get("last_ingested_date")

def _registry_put_date(dataset: str, last_date_str: str) -> None:
    tbl = _dd_table()
    if tbl is None:
        return
    tbl.put_item(Item={"pk": dataset, "last_ingested_date": last_date_str})


# -------------------------------------------------------------------
# NCEI bulk source (recommended; avoids ArcGIS/shapefile dependency)
# -------------------------------------------------------------------

NCEI_BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
DETAILS_RE = re.compile(r"^StormEvents_details-ftp_v1\.0_d(?P<year>\d{4})_c(?P<created>\d{8})\.csv\.gz$")

DEFAULT_WIND_EVENT_TYPES: Set[str] = {
    "THUNDERSTORM WIND",
    "HIGH WIND",
    "STRONG WIND",
    "MARINE THUNDERSTORM WIND",
    "MARINE HIGH WIND",
    "TORNADO",
    "DUST STORM",
}

def _utc_today() -> datetime.date:
    return datetime.now(timezone.utc).date()

def _parse_yyyy_mm_dd(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)

def _fetch_ncei_listing_html() -> str:
    r = requests.get(NCEI_BASE_URL, timeout=60)
    r.raise_for_status()
    return r.text

def _choose_latest_details_file_for_year(html: str, year: int) -> Optional[str]:
    matches: List[Tuple[str, int]] = []
    for fname in set(re.findall(r"StormEvents_details-ftp_v1\.0_d\d{4}_c\d{8}\.csv\.gz", html)):
        m = DETAILS_RE.match(fname)
        if not m:
            continue
        y = int(m.group("year"))
        if y != year:
            continue
        created = int(m.group("created"))
        matches.append((fname, created))
    if not matches:
        return None
    matches.sort(key=lambda t: t[1], reverse=True)
    return matches[0][0]

def _build_daily_from_details_url_chunked(
    url: str,
    state_fips: int,
    gust_cutoff_mph: float,
    event_types: Set[str],
    chunksize: int = 300_000,
) -> pd.DataFrame:
    """
    Faster + memory-safe chunked reader for large NCEI details gz files.

    Output schema:
      county (lowercase), date, year, wind_mph, event_count, severe_gust_58
    """
    usecols = [
        "STATE_FIPS", "CZ_TYPE", "EVENT_TYPE",
        "BEGIN_DATE_TIME", "MAGNITUDE", "MAGNITUDE_TYPE",
        "CZ_NAME",
    ]
    dtype = {
        "STATE_FIPS": "int16",
        "CZ_TYPE": "string",
        "EVENT_TYPE": "string",
        "MAGNITUDE_TYPE": "string",
        "CZ_NAME": "string",
    }

    agg_parts: List[pd.DataFrame] = []

    for chunk in pd.read_csv(
        url,
        compression="gzip",
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = chunk[(chunk["STATE_FIPS"] == state_fips) & (chunk["CZ_TYPE"] == "C")].copy()

        chunk["EVENT_TYPE"] = chunk["EVENT_TYPE"].astype(str).str.upper()
        chunk = chunk[chunk["EVENT_TYPE"].isin(event_types)].copy()
        if chunk.empty:
            continue

        dt = pd.to_datetime(chunk["BEGIN_DATE_TIME"], errors="coerce")
        chunk["date"] = dt.dt.floor("D")
        chunk = chunk.dropna(subset=["date"])
        if chunk.empty:
            continue

        chunk["county"] = (
            chunk["CZ_NAME"].astype(str)
            .str.replace(r"\s+COUNTY$", "", regex=True)
            .str.strip()
            .str.lower()
        )

        # Vectorized wind mph conversion
        chunk["MAGNITUDE"] = pd.to_numeric(chunk["MAGNITUDE"], errors="coerce")
        chunk["wind_mph"] = chunk["MAGNITUDE"]

        mt = chunk["MAGNITUDE_TYPE"].astype(str).str.upper().str.strip()
        chunk.loc[mt == "KT", "wind_mph"] = chunk.loc[mt == "KT", "MAGNITUDE"] * 1.15078
        chunk.loc[mt == "MS", "wind_mph"] = chunk.loc[mt == "MS", "MAGNITUDE"] * 2.23694

        chunk = chunk.dropna(subset=["wind_mph", "county"])
        if chunk.empty:
            continue

        chunk["year"] = chunk["date"].dt.year

        daily = (
            chunk.groupby(["county", "date", "year"], as_index=False)
                 .agg(
                     wind_mph=("wind_mph", "max"),
                     event_count=("EVENT_TYPE", "size"),
                 )
        )
        agg_parts.append(daily)

    if not agg_parts:
        return pd.DataFrame(columns=["county", "date", "year", "wind_mph", "event_count", "severe_gust_58"])

    out = pd.concat(agg_parts, ignore_index=True)
    out = (
        out.groupby(["county", "date", "year"], as_index=False)
           .agg(
               wind_mph=("wind_mph", "max"),
               event_count=("event_count", "sum"),
           )
    )
    out["severe_gust_58"] = (out["wind_mph"] >= gust_cutoff_mph).astype(int)
    return out


def ingest_storm_daily(out: S3Base, region: str, state_fips: str, county_fips: str,
                       start: datetime | None, end: datetime | None) -> None:
    """
    Storm ingestion (daily), writing Hive-style partitions:
      .../dataset=storm/state_fips=XX/county_fips=ALL/granularity=daily/year=YYYY/part.parquet

    Incremental behavior:
      - Uses DynamoDB registry (pk='storm', last_ingested_date) if REGISTRY_TABLE is set.
      - Refreshes recent years using LOOKBACK_DAYS to catch NCEI re-issues.
      - Updates registry ONLY after successful writes.
    """
    dataset = "storm"

    if county_fips.upper() != "ALL":
        raise ValueError("This storm ingestion supports county_fips=ALL only (matches your feature builder).")

    state_i = int(state_fips)

    gust_cutoff = float(os.environ.get("GUST_CUTOFF_MPH", "58"))
    lookback_days = int(os.environ.get("LOOKBACK_DAYS", "45"))
    backfill_start_year = int(os.environ.get("BACKFILL_START_YEAR", "2014"))
    chunksize = int(os.environ.get("STORM_CHUNKSIZE", "300000"))

    last = None
    try:
        last = _registry_get_date(dataset)
    except Exception as e:
        print(f"[STORM] WARN: registry read failed: {e}")

    today = _utc_today()

    if last:
        start_dt = _parse_yyyy_mm_dd(last) - timedelta(days=lookback_days)
        start_year = start_dt.year
    else:
        start_year = backfill_start_year

    years = list(range(start_year, today.year + 1))
    print(f"[STORM] Refreshing years {years} (last_ingested_date={last}, lookback_days={lookback_days})")

    html = _fetch_ncei_listing_html()
    event_types = {
        x.strip().upper()
        for x in os.environ.get("EVENT_TYPES", ",".join(sorted(DEFAULT_WIND_EVENT_TYPES))).split(",")
        if x.strip()
    }

    wrote_any = False

    for year in years:
        fname = _choose_latest_details_file_for_year(html, year)
        if not fname:
            print(f"[STORM] No details file found for year={year}")
            continue

        url = NCEI_BASE_URL + fname
        print(f"[STORM] year={year} file={fname} chunksize={chunksize}")

        daily = _build_daily_from_details_url_chunked(
            url=url,
            state_fips=state_i,
            gust_cutoff_mph=gust_cutoff,
            event_types=event_types,
            chunksize=chunksize,
        )

        if daily.empty:
            print(f"[STORM] year={year}: produced 0 rows (skipping write)")
            continue

        key_prefix = partition_prefix(
            out,
            "storm",
            str(state_i).zfill(2),
            "ALL",
            "daily",
            year=year
        )
        print(f"[STORM] Writing to s3://{out.bucket}/{key_prefix} (rows={len(daily)})")
        write_csv_parquet(region, out, key_prefix, daily, basename="part")
        wrote_any = True

    if wrote_any:
        try:
            _registry_put_date(dataset, today.strftime("%Y-%m-%d"))
            print(f"[STORM] registry updated last_ingested_date={today.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"[STORM] WARN: write succeeded but registry update failed: {e}")
    else:
        print("[STORM] No data written; registry not updated.")


# -------------------------------------------------------------------
# Legacy ArcGIS ingestion (kept for reference) - BUG FIXED:
# Writes partitions PER DAY instead of using start date only.
# -------------------------------------------------------------------

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

def ingest_storm_daily_arcgis_partitioned(out: S3Base, region: str, state_fips: str, county_fips: str,
                                         start: datetime, end: datetime) -> None:
    """
    Legacy ArcGIS ingestion with CORRECT partitioning.
    Writes: .../granularity=daily/year=YYYY/month=MM/day=DD/part.parquet per event day.
    """
    ids = storm_object_ids(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    df = storm_fetch_by_ids(ids)
    if df.empty:
        print("[STORM][ARCGIS] no rows")
        return

    if "BEGIN_DATE" not in df.columns:
        raise ValueError("ArcGIS payload missing BEGIN_DATE")

    df["date"] = pd.to_datetime(df["BEGIN_DATE"], unit="ms", errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    wrote_any = False
    for (y, m, d), part in df.groupby(["year", "month", "day"]):
        key_prefix = partition_prefix(
            out,
            "storm",
            str(int(state_fips)).zfill(2),
            county_fips,
            "daily",
            year=int(y),
            month=int(m),
            day=int(d),
        )
        write_csv_parquet(region, out, key_prefix, part, basename="part")
        wrote_any = True

    if wrote_any:
        _registry_put_date("storm", end.strftime("%Y-%m-%d"))
