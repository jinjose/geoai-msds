import gzip
import re
import pandas as pd
from pathlib import Path
import boto3

STATE_FIPS = 19

# ---- configure these ----
INPUT_GZ = r"./StormEvents_details-ftp_v1.0_d2019_c20260116.csv.gz"  # downloaded file
OUT_PARQUET = "storm_daily_2019.parquet"
S3_BUCKET = "geoai-demo-data"
S3_KEY = "raw/dataset=storm/state_fips=19/county_fips=ALL/granularity=daily/year=2019/storm_daily_2019.parquet"
# -------------------------

WIND_EVENT_TYPES = {
    "THUNDERSTORM WIND",
    "HIGH WIND",
    "STRONG WIND",
    "MARINE THUNDERSTORM WIND",
    "MARINE HIGH WIND",
    "TORNADO",   # optional: tornado may not always have wind magnitude
    "DUST STORM" # optional
}

def normalize_county(name: str) -> str:
    if pd.isna(name):
        return None
    # NCEI CZ_NAME is typically like "POLK" or "POLK COUNTY" depending on source
    s = str(name).strip()
    s = re.sub(r"\s+COUNTY$", "", s, flags=re.I).strip()
    return s.title()  # "Polk"

def to_mph(mag: float, mag_type: str) -> float:
    if pd.isna(mag):
        return None
    mt = (mag_type or "").upper().strip()

    # Many wind records are already mph; sometimes it's "KT" (knots) or "MS" (m/s)
    if mt in ("MPH", ""):
        return float(mag)
    if mt == "KT":
        return float(mag) * 1.15078
    if mt == "MS":
        return float(mag) * 2.23694

    # Unknown magnitude type — keep raw
    return float(mag)

def main():
    # read gz csv
    df = pd.read_csv(INPUT_GZ, compression="gzip", low_memory=False)

    # keep county zones only
    # CZ_TYPE: 'C' county, 'Z' zone
    df = df[(df["STATE_FIPS"] == STATE_FIPS) & (df["CZ_TYPE"] == "C")].copy()

    # wind-related events
    df["EVENT_TYPE"] = df["EVENT_TYPE"].astype(str).str.upper()
    df = df[df["EVENT_TYPE"].isin(WIND_EVENT_TYPES)].copy()

    # datetime
    # BEGIN_DATE_TIME is like "01-JAN-25 00:00:00"
    df["datetime"] = pd.to_datetime(df["BEGIN_DATE_TIME"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # wind magnitude
    df["wind_mph"] = df.apply(lambda r: to_mph(r.get("MAGNITUDE"), r.get("MAGNITUDE_TYPE")), axis=1)
    df = df.dropna(subset=["wind_mph"])

    # county name
    df["county"] = df["CZ_NAME"].apply(normalize_county)
    df = df.dropna(subset=["county"])

    out = df[["county", "datetime", "wind_mph"]].copy()

    # OPTIONAL: reduce volume to daily max per county (recommended)
    out["date"] = out["datetime"].dt.date
    out["year"] = out["datetime"].dt.year
    out = (
        out.groupby(["county", "year", "date"], as_index=False)["wind_mph"].max()
        .assign(datetime=lambda x: pd.to_datetime(x["date"]))
        [["county", "datetime", "wind_mph"]]
    )
    
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET} rows={len(out):,}")

    # upload to S3
    s3 = boto3.client("s3")
    s3.upload_file(OUT_PARQUET, S3_BUCKET, S3_KEY)
    print(f"Uploaded to s3://{S3_BUCKET}/{S3_KEY}")

if __name__ == "__main__":
    main()