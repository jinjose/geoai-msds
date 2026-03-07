import pandas as pd
import requests

import json

def _parse_years_any(years_str: str | None) -> list[int]:
    if years_str is None:
        return []
    s = str(years_str).strip()
    if not s:
        return []
    # Accept JSON list string like "[2014, 2015]" or CSV "2014,2015"
    if s.startswith("["):
        try:
            arr = json.loads(s)
            return [int(x) for x in arr]
        except Exception:
            pass
    return [int(x.strip()) for x in s.split(",") if x.strip()]



# ----------------------------
# Incremental ingestion helpers (DynamoDB)
# ----------------------------
import os
import boto3
from datetime import datetime

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

def _registry_get_year(dataset: str):
    tbl = _dd_table()
    if tbl is None:
        return None
    resp = tbl.get_item(Key={"pk": dataset})
    return (resp.get("Item") or {}).get("last_ingested_year")

def _registry_put_year(dataset: str, last_year: int):
    tbl = _dd_table()
    if tbl is None:
        return
    tbl.put_item(Item={"pk": dataset, "last_ingested_year": int(last_year)})


def fetch_yield_year(api_key: str, year: int, state_fips: str, county_fips: str) -> pd.DataFrame:
    print(f"[YIELD] Fetching year={year} state={state_fips} county={county_fips}")
    base = "https://quickstats.nass.usda.gov/api/api_GET/"
    params = {
        "key": api_key,
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "commodity_desc": "CORN",
        "class_desc": "GRAIN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "agg_level_desc": "COUNTY",
        "year": str(year),
        "state_fips_code": str(state_fips).zfill(2),
        "format": "JSON",
    }
    if county_fips.upper() != "ALL":
        params["county_code"] = str(county_fips).zfill(3)

    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    rows = data.get("data", [])
    print(f"[YIELD] Raw rows returned: {len(rows)}")
    if not rows:
        print(f"[YIELD] No data for year={year}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"[YIELD] Processed rows: {len(df)}")

    if "Value" in df.columns:
        df["yield_bu_acre"] = df["Value"].astype(str).str.replace(",", "", regex=False)
        df["yield_bu_acre"] = pd.to_numeric(df["yield_bu_acre"], errors="coerce")

    keep = [c for c in ["year","state_fips_code","county_code","county_name","yield_bu_acre"] if c in df.columns]
    print(f"[YIELD] Final columns: {keep}")
    return df[keep].copy()

def ingest_yield(api_key: str, years_csv: str | None, state_fips: str, county_fips: str) -> pd.DataFrame:
    """
    Yield ingestion (yearly via QuickStats).

    If years_csv is empty/None and REGISTRY_TABLE is set, this will do incremental ingestion:
      years = (last_ingested_year + 1) .. (current_year - 1)

    Registry is updated only after we successfully fetched at least one year with rows.
    """
    dataset = "yield"

    years = []
    if years_csv and str(years_csv).strip():
        years = _parse_years_any(years_csv)
    else:
        last_year = _registry_get_year(dataset)
        max_year = datetime.utcnow().year - 1  # last complete year
        if last_year is None:
            backfill_year = int(os.environ.get("BACKFILL_START_YEAR", "2013"))
            start_year = backfill_year
        else:
            start_year = int(last_year) + 1
        years = list(range(start_year, max_year + 1))
        print(f"[YIELD] incremental years={years} (last={last_year})")

    print(f"[YIELD] Years requested: {years}")

    if not years:
        print("[YIELD] No years to ingest — skipping")
        return pd.DataFrame()

    frames = [fetch_yield_year(api_key, y, state_fips, county_fips) for y in years]
    frames = [f for f in frames if f is not None and not f.empty]

    if not frames:
        print("[YIELD] No data returned for requested years — registry not updated")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    print(f"[YIELD] Final DataFrame rows: {len(out)}")

    _registry_put_year(dataset, max(years))
    print(f"[YIELD] registry updated last_ingested_year={max(years)}")

    return out