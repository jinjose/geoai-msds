import argparse, json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

import ee
from google.oauth2 import service_account


# ---------------------------
# Secrets + GEE init
# ---------------------------
def secrets_json(secret_id: str, region: str) -> dict:
    sm = boto3.client("secretsmanager", region_name=region)
    return json.loads(sm.get_secret_value(SecretId=secret_id)["SecretString"])

def secrets_text(secret_id: str, region: str) -> str:
    sm = boto3.client("secretsmanager", region_name=region)
    return sm.get_secret_value(SecretId=secret_id)["SecretString"].strip()

def init_gee(sa_secret_id: str, region: str) -> None:
    sa_info = secrets_json(sa_secret_id, region)
    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/earthengine"],
    )
    ee.Initialize(creds)


# ---------------------------
# Time utilities
# ---------------------------
def parse_dt(s: str) -> datetime:
    # YYYY-MM-DD OR YYYY-MM-DDTHH:MM:SSZ
    if len(s) == 10:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

def floor_to(dt: datetime, gran: str) -> datetime:
    if gran == "hourly":
        return dt.replace(minute=0, second=0, microsecond=0)
    if gran == "daily":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if gran == "weekly":
        d = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return d - timedelta(days=d.weekday())
    if gran == "monthly":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if gran == "yearly":
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(gran)

def step(dt: datetime, gran: str) -> datetime:
    if gran == "hourly":
        return dt + timedelta(hours=1)
    if gran == "daily":
        return dt + timedelta(days=1)
    if gran == "weekly":
        return dt + timedelta(days=7)
    if gran == "monthly":
        y, m = dt.year, dt.month + 1
        if m == 13:
            y, m = y + 1, 1
        return dt.replace(year=y, month=m, day=1)
    if gran == "yearly":
        return dt.replace(year=dt.year + 1, month=1, day=1)
    raise ValueError(gran)

def iter_windows(start: datetime, end: datetime, gran: str):
    cur = floor_to(start, gran)
    while cur < end:
        nxt = step(cur, gran)
        yield max(cur, start), min(nxt, end)
        cur = nxt


# ---------------------------
# S3 write both CSV + Parquet
# ---------------------------
@dataclass(frozen=True)
class S3Base:
    bucket: str
    prefix: str

def parse_s3(uri: str) -> S3Base:
    assert uri.startswith("s3://")
    rest = uri[5:]
    b, _, p = rest.partition("/")
    if p and not p.endswith("/"):
        p += "/"
    return S3Base(bucket=b, prefix=p)

def s3_upload(local_path: str, bucket: str, key: str, region: str):
    boto3.client("s3", region_name=region).upload_file(local_path, bucket, key)

def partition_prefix(base: S3Base, dataset: str, gran: str, wstart: datetime) -> str:
    parts = [
        base.prefix.rstrip("/"),
        f"dataset={dataset}",
        f"granularity={gran}",
        f"year={wstart.year:04d}",
        f"month={wstart.month:02d}",
    ]
    if gran in ("daily", "weekly", "hourly"):
        parts.append(f"day={wstart.day:02d}")
    if gran == "hourly":
        parts.append(f"hour={wstart.hour:02d}")
    return "/".join([p for p in parts if p]) + "/"

def write_csv_parquet(df: pd.DataFrame, base: S3Base, region: str,
                      dataset: str, gran: str, wstart: datetime):
    pref = partition_prefix(base, dataset, gran, wstart)
    stamp = wstart.strftime("%Y%m%dT%H%M%SZ") if gran == "hourly" else wstart.strftime("%Y%m%d")

    csv_key = f"{pref}data_{stamp}.csv"
    pq_key  = f"{pref}data_{stamp}.parquet"

    csv_path = f"/tmp/{dataset}_{stamp}.csv"
    pq_path  = f"/tmp/{dataset}_{stamp}.parquet"

    df.to_csv(csv_path, index=False)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, pq_path, compression="snappy")

    s3_upload(csv_path, base.bucket, csv_key, region)
    s3_upload(pq_path, base.bucket, pq_key, region)


# ---------------------------
# Common geo: Iowa counties
# ---------------------------
def counties_fc(state_fips: str, county_fips: str | None = None):
    """
    state_fips: 2-digit string (e.g. '19' for Iowa)
    county_fips: optional 3-digit county code (e.g. '001')
    """

    fc = ee.FeatureCollection("TIGER/2018/Counties") \
            .filter(ee.Filter.eq("STATEFP", state_fips))

    if county_fips:
        # Combine state + county → full GEOID
        geoid = state_fips + county_fips.zfill(3)
        fc = fc.filter(ee.Filter.eq("GEOID", geoid))

    return fc


# ---------------------------
# ERA5-Land: daily + hourly
# Schema:
# county_fips,county_name,date,evapotranspiration,month,rainfall,soil_moisture,temperature,year
# For hourly, we also include "hour" column (useful downstream).
# ---------------------------
ERA5_DAILY_VARS = {
    "temperature": "temperature_2m",
    "rainfall": "total_precipitation_sum",
    "evapotranspiration": "total_evaporation_sum",
    "soil_moisture": "volumetric_soil_water_layer_1",
}

# Hourly variables differ (often not *_sum). Adjust as needed per EE collection.
ERA5_HOURLY_VARS = {
    "temperature": "temperature_2m",
    "rainfall": "total_precipitation",         # hourly precipitation (not sum)
    "evapotranspiration": "evaporation",       # hourly evaporation (if available)
    "soil_moisture": "volumetric_soil_water_layer_1",
}

def era5_reduce_over_counties(image: ee.Image, counties: ee.FeatureCollection, props_base: dict, varmap: dict):
    img = image.select(list(varmap.values()))

    def per_county(county):
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=county.geometry(),
            scale=10000,
            bestEffort=True,
            maxPixels=1e13,
        )
        props = dict(props_base)
        props.update({
            "county_fips": county.get("GEOID"),
            "county_name": county.get("NAME"),
            "temperature": stats.get(varmap["temperature"]),
            "rainfall": stats.get(varmap["rainfall"]),
            "evapotranspiration": stats.get(varmap["evapotranspiration"]),
            "soil_moisture": stats.get(varmap["soil_moisture"]),
        })
        return ee.Feature(None, props)

    return counties.map(per_county)

def fetch_era5(window_start: datetime, window_end: datetime, granularity: str,counties) -> pd.DataFrame:
  
    if granularity == "daily":
        start = ee.Date(window_start.strftime("%Y-%m-%d"))
        end = ee.Date(window_end.strftime("%Y-%m-%d"))
        coll = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(start, end)

        def per_image(img):
            props_base = {
                "date": img.date().format("YYYY-MM-dd"),
                "year": window_start.year,
                "month": window_start.month,
            }
            return era5_reduce_over_counties(img, counties, props_base, ERA5_DAILY_VARS)

        fc = coll.map(per_image).flatten()

    elif granularity == "hourly":
        # Hourly collection
        # If this exact ID differs in your EE account, change it here.
        start = ee.Date(window_start.isoformat().replace("+00:00", "Z"))
        end = ee.Date(window_end.isoformat().replace("+00:00", "Z"))
        coll = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start, end)

        def per_image(img):
            props_base = {
                "date": img.date().format("YYYY-MM-dd"),
                "year": window_start.year,
                "month": window_start.month
                #"hour": img.date().format("HH"),   # string hour; we cast later

            }
            return era5_reduce_over_counties(img, counties, props_base, ERA5_HOURLY_VARS)

        fc = coll.map(per_image).flatten()

    else:
        raise ValueError("ERA5 supported only for daily/hourly in this runner")

    rows = [f["properties"] for f in fc.getInfo()["features"]]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    # Keep your daily schema; for hourly we add hour.
    base_cols = [
        "county_fips","county_name","date","evapotranspiration","month",
        "rainfall","soil_moisture","temperature","year"
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None

    # if granularity == "hourly":
    #     df["hour"] = pd.to_numeric(df.get("hour"), errors="coerce").astype("Int64")
    #     return df[base_cols + ["hour"]]

    return df[base_cols]


# ---------------------------
# NDVI (MOD13Q1) – daily windows supported; hourly policy needed
# Schema: NDVI,county_name,date,geoid,year
# ---------------------------
def corn_mask_for_year(year: int):
    cdl = ee.ImageCollection("USDA/NASS/CDL").filterDate(f"{year}-01-01", f"{year}-12-31").first().select("cropland")
    return cdl.eq(1)

def fetch_ndvi_daily_like(window_start: datetime, window_end: datetime,counties) -> pd.DataFrame:
    year = window_start.year
    start = window_start.strftime("%Y-%m-%d")
    end = window_end.strftime("%Y-%m-%d")
    corn_mask = corn_mask_for_year(year)

    coll = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start, end).select("NDVI")

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
            return ee.Feature(None, {
                "date": image.date().format("YYYY-MM-dd"),
                "year": year,
                "county_name": county.get("NAME"),
                "geoid": county.get("GEOID"),
                "NDVI": stats.get("NDVI"),
            })

        return counties.map(per_county)

    fc = coll.map(per_image).flatten()
    rows = [f["properties"] for f in fc.getInfo()["features"]]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["year"] = df["year"].astype(int)
    df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce") / 10000.0
    return df[["NDVI","county_name","date","geoid","year"]]

def fetch_ndvi(window_start: datetime,
               window_end: datetime,
               granularity: str,
               hourly_policy: str,
               counties) -> pd.DataFrame:

    if granularity == "daily":
        return fetch_ndvi_daily_like(window_start, window_end, counties)

    if granularity == "hourly":
        if hourly_policy == "skip":
            return pd.DataFrame(columns=["NDVI","county_name","date","geoid","year","hour"])

        day_start = window_start.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        df = fetch_ndvi_daily_like(day_start, day_end, counties)

        if df.empty:
            return df

        df["hour"] = window_start.hour
        return df[["NDVI","county_name","date","geoid","year","hour"]]

    raise ValueError("NDVI supported only for daily/hourly")


# ---------------------------
# Yield (still yearly; you can schedule yearly/monthly pulls)
# ---------------------------
QUICKSTATS_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
YIELD_COLUMNS = [
    "Program","Year","Period","Week Ending","Geo Level","State","State ANSI","Ag District","Ag District Code",
    "County","County ANSI","Zip Code","Region","watershed_code","Watershed","Commodity","Data Item","Domain",
    "Domain Category","Value","CV (%)"
]

def fetch_yield_year(api_key: str,
                     year: int,
                     state_name: str,
                     county_ansi: str | None = None) -> pd.DataFrame:

    params = {
        "key": api_key,
        "format": "JSON",
        "year": str(year),
        "state_name": state_name,
        "commodity_desc": "CORN"
    }

    if county_ansi:
        params["county_code"] = county_ansi.zfill(3)

    r = requests.get(QUICKSTATS_URL, params=params, timeout=60)
    r.raise_for_status()

    data = r.json().get("data", [])
    df = pd.DataFrame(data)

    if df.empty:
        return df

    mapping = {
        "program": "Program",
        "year": "Year",
        "period": "Period",
        "week_ending": "Week Ending",
        "agg_level_desc": "Geo Level",
        "state_name": "State",
        "state_ansi": "State ANSI",
        "asd_desc": "Ag District",
        "asd_code": "Ag District Code",
        "county_name": "County",
        "county_ansi": "County ANSI",
        "zip_5": "Zip Code",
        "region_desc": "Region",
        "watershed_code": "watershed_code",
        "watershed_desc": "Watershed",
        "commodity_desc": "Commodity",
        "short_desc": "Data Item",
        "domain_desc": "Domain",
        "domaincat_desc": "Domain Category",
        "value": "Value",
        "cv_%": "CV (%)",
    }

    for src, dst in mapping.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    for c in YIELD_COLUMNS:
        if c not in df.columns:
            df[c] = None

    return df[YIELD_COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aws-region", required=True)
    ap.add_argument("--out-s3", required=True, help="s3://bucket/prefix")
    ap.add_argument("--dataset", required=True, choices=["era5","ndvi","yield"])
    ap.add_argument("--state-fips", required=True,
                help="2-digit state FIPS (e.g. 19 for Iowa)")
    ap.add_argument("--county-fips", default="ALL",
                help="3-digit county FIPS (e.g. 001). Use ALL for all counties.")

    ap.add_argument("--granularity", required=False, default="daily", choices=["hourly","daily","weekly","monthly","yearly"])
    ap.add_argument("--start", required=False)
    ap.add_argument("--end", required=False)

    ap.add_argument("--gee-secret-id", required=False)
    ap.add_argument("--nass-secret-id", required=False)
    ap.add_argument("--year", type=int, required=False)

    ap.add_argument("--ndvi-hourly-policy", choices=["skip","carryforward"], default="carryforward")

    args = ap.parse_args()
    out = parse_s3(args.out_s3)
    county_fips = None if args.county_fips.upper() == "ALL" else args.county_fips

    counties = counties_fc(args.state_fips, county_fips)

    if args.dataset in ("era5","ndvi"):
        if not args.gee_secret_id:
            raise SystemExit("--gee-secret-id required for era5/ndvi")
        if not args.start or not args.end:
            raise SystemExit("--start and --end required for era5/ndvi")
        init_gee(args.gee_secret_id, args.aws_region)

        start = parse_dt(args.start)
        end = parse_dt(args.end)

        # For now support hourly/daily windows for era5/ndvi
        if args.granularity not in ("hourly","daily"):
            raise SystemExit("For now, use --granularity hourly or daily for era5/ndvi")

        for wstart, wend in iter_windows(start, end, args.granularity):
            if args.dataset == "era5":
                df = fetch_era5(wstart, wend, args.granularity,counties)
            else:
                df = fetch_ndvi(wstart, wend, args.granularity, args.ndvi_hourly_policy,counties)

            if df is None or df.empty:
                continue
            write_csv_parquet(df, out, args.aws_region, args.dataset, args.granularity, wstart)

        print("Done")

    else:
        if not args.nass_secret_id or not args.year:
            raise SystemExit("--nass-secret-id and --year required for yield")
        api_key = secrets_text(args.nass_secret_id, args.aws_region)
        #df = fetch_yield_year(api_key, args.year, state="IOWA")
        county_ansi = None if args.county_fips.upper() == "ALL" else args.county_fips

        # Optional: map state_fips to state_name if needed
        # For demo, you can pass state_name separately via CLI if preferred.
        STATE_FIPS_TO_NAME = {
            "01": "ALABAMA",
            "02": "ALASKA",
            "04": "ARIZONA",
            "05": "ARKANSAS",
            "06": "CALIFORNIA",
            "08": "COLORADO",
            "09": "CONNECTICUT",
            "10": "DELAWARE",
            "11": "DISTRICT OF COLUMBIA",
            "12": "FLORIDA",
            "13": "GEORGIA",
            "15": "HAWAII",
            "16": "IDAHO",
            "17": "ILLINOIS",
            "18": "INDIANA",
            "19": "IOWA",
            "20": "KANSAS",
            "21": "KENTUCKY",
            "22": "LOUISIANA",
            "23": "MAINE",
            "24": "MARYLAND",
            "25": "MASSACHUSETTS",
            "26": "MICHIGAN",
            "27": "MINNESOTA",
            "28": "MISSISSIPPI",
            "29": "MISSOURI",
            "30": "MONTANA",
            "31": "NEBRASKA",
            "32": "NEVADA",
            "33": "NEW HAMPSHIRE",
            "34": "NEW JERSEY",
            "35": "NEW MEXICO",
            "36": "NEW YORK",
            "37": "NORTH CAROLINA",
            "38": "NORTH DAKOTA",
            "39": "OHIO",
            "40": "OKLAHOMA",
            "41": "OREGON",
            "42": "PENNSYLVANIA",
            "44": "RHODE ISLAND",
            "45": "SOUTH CAROLINA",
            "46": "SOUTH DAKOTA",
            "47": "TENNESSEE",
            "48": "TEXAS",
            "49": "UTAH",
            "50": "VERMONT",
            "51": "VIRGINIA",
            "53": "WASHINGTON",
            "54": "WEST VIRGINIA",
            "55": "WISCONSIN",
            "56": "WYOMING",

            # Territories (if ever needed)
            "60": "AMERICAN SAMOA",
            "66": "GUAM",
            "69": "NORTHERN MARIANA ISLANDS",
            "72": "PUERTO RICO",
            "78": "VIRGIN ISLANDS"
            }
        state_name = STATE_FIPS_TO_NAME.get(args.state_fips)

        if not state_name:
            raise SystemExit("Unsupported state_fips for yield mapping")

        df = fetch_yield_year(api_key,
                            args.year,
                            state_name=state_name,
                            county_ansi=county_ansi)

        if df is None or df.empty:
            print("No yield returned")
            return
        # store yield as yearly partition
        wstart = datetime(args.year, 1, 1, tzinfo=timezone.utc)
        write_csv_parquet(df, out, args.aws_region, "yield", "yearly", wstart)
        print("Done")


if __name__ == "__main__":
    main()