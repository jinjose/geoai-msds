#!/usr/bin/env python3
"""
download_make_storm_daily_to_s3.py

Downloads yearly StormEvents "details" bulk files from:
  https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

File naming pattern (examples):
  StormEvents_details-ftp_v1.0_d2019_c20260116.csv.gz
  where d = data year, c = creation date (YYYYMMDD)

For each requested year, this script:
  - picks the newest "cYYYYMMDD" file for that year
  - downloads .csv.gz
  - filters by STATE_FIPS and county zones (CZ_TYPE == 'C')
  - filters to wind-related EVENT_TYPEs (configurable)
  - converts magnitude units to mph
  - aggregates to county/day: max wind_mph + event_count
  - creates severe_gust flag by cutoff (default 58 mph)
  - standardizes schema via generic_storm_standardizer.py
  - writes parquet and uploads to S3 in:
      {s3_prefix}/state_fips={state}/county_fips=ALL/granularity=daily/year={year}/storm_daily_{year}.parquet

Requirements:
  pip install pandas pyarrow boto3 requests

Usage:
  python download_make_storm_daily_to_s3.py \
    --years 2013-2025 \
    --state-fips 19 \
    --s3-bucket geoai-demo-data \
    --s3-prefix raw/dataset=storm \
    --workdir ./_storm_work

Notes:
  - AWS credentials should be available via env/role (SageMaker/ECS/CLI).
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests
import boto3

# ---- import your standardizer (same folder or PYTHONPATH) ----
# generic_storm_standardizer.py defines: standardize_storm_df, StandardizeOptions
from generic_storm_standardizer import standardize_storm_df, StandardizeOptions


BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
# matches: StormEvents_details-ftp_v1.0_d2019_c20260116.csv.gz
DETAILS_RE = re.compile(
    r'^StormEvents_details-ftp_v1\.0_d(?P<year>\d{4})_c(?P<created>\d{8})\.csv\.gz$'
)

DEFAULT_WIND_EVENT_TYPES = {
    "THUNDERSTORM WIND",
    "HIGH WIND",
    "STRONG WIND",
    "MARINE THUNDERSTORM WIND",
    "MARINE HIGH WIND",
    # Optional includes (keep if your model expects them):
    "TORNADO",
    "DUST STORM",
}


def parse_years(spec: str) -> List[int]:
    """
    Accept:
      "2019"
      "2013,2014,2015"
      "2013-2025"
    """
    spec = spec.strip()
    if "," in spec:
        out = []
        for part in spec.split(","):
            out.extend(parse_years(part))
        return sorted(set(out))
    if "-" in spec:
        a, b = spec.split("-", 1)
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return [int(spec)]


def normalize_county(name: str) -> Optional[str]:
    if pd.isna(name):
        return None
    s = str(name).strip()
    s = re.sub(r"\s+COUNTY$", "", s, flags=re.I).strip()
    # keep title-case here; standardizer will lower() it to canonical
    return s.title()


def to_mph(mag: float, mag_type: str) -> Optional[float]:
    if pd.isna(mag):
        return None
    mt = (mag_type or "").upper().strip()

    if mt in ("MPH", ""):
        return float(mag)
    if mt == "KT":
        return float(mag) * 1.15078
    if mt == "MS":
        return float(mag) * 2.23694

    # unknown unit: keep numeric value as-is (better than dropping)
    try:
        return float(mag)
    except Exception:
        return None


def fetch_directory_listing_html(session: requests.Session) -> str:
    r = session.get(BASE_URL, timeout=60)
    r.raise_for_status()
    return r.text


def choose_latest_file_for_year(html: str, year: int) -> Optional[str]:
    """
    Parse directory listing HTML and find the newest file for the given year
    based on the creation date cYYYYMMDD in the filename.
    """
    matches: List[Tuple[str, int]] = []
    for fname in set(re.findall(r'StormEvents_details-ftp_v1\.0_d\d{4}_c\d{8}\.csv\.gz', html)):
        m = DETAILS_RE.match(fname)
        if not m:
            continue
        y = int(m.group("year"))
        if y != year:
            continue
        c = int(m.group("created"))
        matches.append((fname, c))

    if not matches:
        return None

    matches.sort(key=lambda t: t[1], reverse=True)  # newest creation date first
    return matches[0][0]


def download_file(session: requests.Session, url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def build_daily_df_from_details_gz(
    gz_path: Path,
    state_fips: int,
    wind_event_types: set[str],
    gust_cutoff_mph: float,
) -> pd.DataFrame:
    """
    Reads the NCEI 'details' file and returns canonical daily dataframe:
      county, date, year, wind_mph, event_count, severe_gust_58
    """
    df = pd.read_csv(gz_path, compression="gzip", low_memory=False)

    # Filter to the target state and county zones
    df = df[(df["STATE_FIPS"] == state_fips) & (df["CZ_TYPE"] == "C")].copy()

    # Filter to wind-related events
    df["EVENT_TYPE"] = df["EVENT_TYPE"].astype(str).str.upper()
    df = df[df["EVENT_TYPE"].isin(wind_event_types)].copy()

    # Parse begin datetime
    # Example: "01-JAN-19 00:00:00"
    dt = pd.to_datetime(df["BEGIN_DATE_TIME"], errors="coerce")
    df = df.assign(datetime=dt).dropna(subset=["datetime"])

    # Convert magnitude to mph
    df["wind_mph"] = df.apply(
        lambda r: to_mph(r.get("MAGNITUDE"), r.get("MAGNITUDE_TYPE")),
        axis=1
    )
    df = df.dropna(subset=["wind_mph"])

    # County normalize
    df["county"] = df["CZ_NAME"].apply(normalize_county)
    df = df.dropna(subset=["county"])

    # Aggregate to daily max + count
    df["date"] = df["datetime"].dt.floor("D")
    df["year"] = df["date"].dt.year

    daily = (
        df.groupby(["county", "date", "year"], as_index=False)
          .agg(
              wind_mph=("wind_mph", "max"),
              event_count=("EVENT_TYPE", "size"),
          )
    )
    daily["severe_gust_58"] = (daily["wind_mph"] >= gust_cutoff_mph).astype(int)

    # Standardizer expects "county" + ("date" or "datetime") + wind_mph; we already have canon cols.
    # We'll keep these columns; standardizer will normalize county text and ensure types.
    return daily


def s3_key_for_year(s3_prefix: str, state_fips: int, year: int) -> str:
    prefix = s3_prefix.strip("/")

    return (
        f"{prefix}/"
        f"state_fips={state_fips}/"
        f"county_fips=ALL/"
        f"granularity=daily/"
        f"year={year}/"
        f"storm_daily_{year}.parquet"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--years", required=True, help="e.g. 2013-2025 or 2019 or 2013,2014,2015")
    p.add_argument("--state-fips", type=int, default=19)
    p.add_argument("--gust-cutoff", type=float, default=58.0)
    p.add_argument("--s3-bucket", required=True)
    p.add_argument("--s3-prefix", required=True, help="e.g. raw/dataset=storm")
    p.add_argument("--workdir", default="./_storm_work")
    p.add_argument("--keep-downloads", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--event-types",
        default=",".join(sorted(DEFAULT_WIND_EVENT_TYPES)),
        help="Comma-separated EVENT_TYPE list to keep"
    )
    args = p.parse_args()

    years = parse_years(args.years)
    wind_event_types = {x.strip().upper() for x in args.event_types.split(",") if x.strip()}

    workdir = Path(args.workdir)
    downloads_dir = workdir / "downloads"
    outputs_dir = workdir / "outputs"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    html = fetch_directory_listing_html(session)

    s3 = boto3.client("s3")

    for year in years:
        fname = choose_latest_file_for_year(html, year)
        if not fname:
            print(f"[WARN] No matching file found for year={year} in directory listing.")
            continue

        url = BASE_URL + fname
        gz_path = downloads_dir / fname
        out_path = outputs_dir / f"storm_daily_{year}.parquet"
        s3_key = s3_key_for_year(args.s3_prefix, args.state_fips, year)

        print(f"\nYear={year}")
        print(f"  Chosen: {fname}")
        print(f"  URL:    {url}")
        print(f"  Local:  {gz_path}")
        print(f"  Out:    {out_path}")
        print(f"  S3:     s3://{args.s3_bucket}/{s3_key}")

        if args.dry_run:
            continue

        # Download
        if not gz_path.exists():
            print("  Downloading...")
            download_file(session, url, gz_path)
        else:
            print("  Using cached download...")

        # Build daily dataset
        print("  Building daily dataset...")
        daily = build_daily_df_from_details_gz(
            gz_path=gz_path,
            state_fips=args.state_fips,
            wind_event_types=wind_event_types,
            gust_cutoff_mph=args.gust_cutoff,
        )

        # Standardize to canonical schema (your existing normalizer)
        # (Will also lower-case county, ensure types, recompute/validate severe flag if needed.)
        opts = StandardizeOptions(gust_cutoff_mph=args.gust_cutoff)
        daily_std = standardize_storm_df(daily, opts)

        # Write parquet
        print(f"  Writing parquet rows={len(daily_std):,} ...")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        daily_std.to_parquet(out_path, index=False)

        # Upload to S3
        print("  Uploading to S3...")
        s3.upload_file(str(out_path), args.s3_bucket, s3_key)
        print("  Done.")

        # Cleanup downloads if requested
        if (not args.keep_downloads) and gz_path.exists():
            try:
                gz_path.unlink()
            except Exception as e:
                print(f"  [WARN] could not delete {gz_path}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()