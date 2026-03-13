import argparse
import os
from .env import load_settings
from .gee import init_gee
from .secrets import secrets_text
from .timeutils import parse_dt, iter_windows
from .s3io import S3Base, write_csv_parquet, partition_prefix

from .datasets.ndvi import ingest_ndvi
from .datasets.era5 import ingest_era5
from .datasets.storm import ingest_storm_daily
from .datasets.yield_nass import ingest_yield

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["ndvi", "era5", "storm", "yield", "all"])
    args = p.parse_args()

    s = load_settings()

    # ---------------------------------------------------
    # Determine ingestion window
    # ---------------------------------------------------
    if s.ingest_start and s.ingest_end:
        start = parse_dt(s.ingest_start)
        end = parse_dt(s.ingest_end)
        print(f"[CLI] Manual window={start.date()} → {end.date()}")
    else:
        start = None
        end = None
        print("[CLI] Incremental mode (window determined by DynamoDB)")

    out = S3Base(bucket=s.data_bucket, prefix=s.raw_prefix.rstrip("/") + "/")

    print(f"[CLI] Dataset={args.dataset}")
    print(f"[CLI] Output bucket={s.data_bucket}")
    print(f"[CLI] REGISTRY_TABLE: {os.environ.get('REGISTRY_TABLE')}")

    # ---------------------------------------------------
    # GEE init (only when needed)
    # ---------------------------------------------------
    if args.dataset in ("ndvi", "era5", "all"):
        print(f"[CLI] Initializing GEE with secret ID={s.gee_sa_secret_id}")
        init_gee(s.gee_sa_secret_id, s.aws_region)

    # ---------------------------------------------------
    # NDVI
    # ---------------------------------------------------
    if args.dataset in ("ndvi", "all"):
        print("[CLI] Ingesting NDVI data")
        df = ingest_ndvi(s.state_fips, s.county_fips, start, end, s.ndvi_granularity)

        if df is not None and not df.empty:
            key_prefix = partition_prefix(
                out, "ndvi", s.state_fips, s.county_fips,
                s.ndvi_granularity,
                year=df["year"].iloc[0] if "year" in df.columns else None
            )
            write_csv_parquet(s.aws_region, out, key_prefix, df, basename="part")
            print("[CLI] Finished NDVI ingest")

    # ---------------------------------------------------
    # ERA5
    # ---------------------------------------------------
    if args.dataset in ("era5", "all"):
        print("[CLI] Ingesting ERA5 data")
        g = s.era5_granularity.lower()

        if start is None or end is None:
            # incremental mode
            df = ingest_era5(s.state_fips, s.county_fips, None, None, g)
            if df is not None and not df.empty:
                key_prefix = partition_prefix(
                    out, "era5", s.state_fips, s.county_fips, g,
                    year=df["year"].iloc[0] if "year" in df.columns else None
                )
                write_csv_parquet(s.aws_region, out, key_prefix, df, basename="part")
                print("[CLI] Finished ERA5 incremental ingest")
        else:
            # manual window mode
            for ws, we in iter_windows(start, end, "daily"):
                df = ingest_era5(s.state_fips, s.county_fips, ws, we, g)
                if df is None or df.empty:
                    continue

                df["year"] = ws.year
                df["month"] = ws.month
                df["day"] = ws.day

                key_prefix = partition_prefix(
                    out, "era5", s.state_fips, s.county_fips, g,
                    year=ws.year, month=ws.month, day=ws.day
                )
                write_csv_parquet(s.aws_region, out, key_prefix, df, basename="part")
                print(f"[CLI] Finished ERA5 ingest {ws.date()} → {we.date()}")

    # ---------------------------------------------------
    # STORM
    # ---------------------------------------------------
    if args.dataset in ("storm", "all"):
        if start is None or end is None:
            ingest_storm_daily(out, s.aws_region, s.state_fips, s.county_fips, None, None)
            print("[CLI] Finished storm incremental ingest")
        else:
            for ws, we in iter_windows(start, end, "daily"):
                ingest_storm_daily(out, s.aws_region, s.state_fips, s.county_fips, ws, we)
                print(f"[CLI] Finished storm ingest {ws.date()} → {we.date()}")

    # ---------------------------------------------------
    # YIELD
    # ---------------------------------------------------
    if args.dataset in ("yield", "all"):
        api_key = secrets_text(s.nass_api_key_secret_id, s.aws_region)
        df = ingest_yield(api_key, s.yield_years, s.state_fips, s.county_fips)

        if df is not None and not df.empty:
            key_prefix = partition_prefix(
                out, "yield", s.state_fips, s.county_fips, "yearly",
                year=df["year"].max()
            )
            write_csv_parquet(s.aws_region, out, key_prefix, df, basename="part")
            print("[CLI] Finished yield ingest")
        

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[CLI] Error occurred: {e}")
        raise