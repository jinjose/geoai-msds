import argparse
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
    start = parse_dt(s.ingest_start)
    end   = parse_dt(s.ingest_end)    
    out = S3Base(bucket=s.data_bucket, prefix=s.raw_prefix.rstrip("/") + "/")

    print(f"[CLI] Dataset={args.dataset}")
    print(f"[CLI] Window={start.date()} → {end.date()}")
    print(f"[CLI] Output bucket={s.data_bucket}")

    if args.dataset in ("ndvi", "era5", "all"):
        print(f"[CLI] Initializing GEE with secret ID={s.gee_sa_secret_id}")
        init_gee(s.gee_sa_secret_id, s.aws_region)
        p

    if args.dataset in ("ndvi", "all"):
        print(f"[CLI] Ingesting NDVI data")
        df = ingest_ndvi(s.state_fips, s.county_fips, start, end, s.ndvi_granularity)
        key_prefix = partition_prefix(out, "ndvi", s.state_fips, s.county_fips, s.ndvi_granularity,
                                      year=start.year, month=start.month, day=start.day)
        write_csv_parquet(s.aws_region, out, key_prefix, df, basename="part")
        print(f"[CLI] Finished NDVI ingest")

    if args.dataset in ("era5", "all"):
        print(f"[CLI] Ingesting ERA5 data")
        g = s.era5_granularity.lower()
        if g in ("daily", "hourly"):
            # ✅ Chunk into daily windows so features << 5000 per request
            for ws, we in iter_windows(start, end, "daily"):
                df = ingest_era5(s.state_fips, s.county_fips, ws, we, g)
                    # harden schema (permanent)
                df["year"] = ws.year
                df["month"] = ws.month
                df["day"] = ws.day
                key_prefix = partition_prefix(
                    out, "era5", s.state_fips, s.county_fips, g,
                    year=ws.year, month=ws.month, day=ws.day
                )
                write_csv_parquet(s.aws_region, out, key_prefix, df, basename="part")
                print(f"[CLI] Finished ERA5 ingest for window {ws.date()} → {we.date()}")
        elif g == "yearly":
            df = ingest_era5(s.state_fips, s.county_fips, start, end, g)
            # ✅ yearly should NOT write under month/day partitions
            key_prefix = partition_prefix(
                out, "era5", s.state_fips, s.county_fips, g,
                year=start.year
            )
            write_csv_parquet(s.aws_region, out, key_prefix, df, basename="part")
            print(f"[CLI] Finished ERA5 ingest for window {start.date()} → {end.date()}")
        else:
            raise ValueError("Unsupported ERA5_GRANULARITY. Use daily/hourly/yearly.")

    if args.dataset in ("storm", "all"):
        for ws, we in iter_windows(start, end, "daily"):
            ingest_storm_daily(out, s.aws_region, s.state_fips, s.county_fips, ws, we)
            print(f"[CLI] Finished storm ingest for window {ws.date()} → {we.date()}")

    if args.dataset in ("yield", "all"):
        if not s.yield_years.strip():
            raise ValueError("YIELD_YEARS env is required for yield ingestion (example: 2024,2025)")
        api_key = secrets_text(s.nass_api_key_secret_id, s.aws_region)
        df = ingest_yield(api_key, s.yield_years, s.state_fips, s.county_fips)
        key_prefix = partition_prefix(out, "yield", s.state_fips, s.county_fips, "yearly",
                                      year=start.year)
        write_csv_parquet(s.aws_region, out, key_prefix, df, basename="part")
        print(f"[CLI] Finished yield ingest for years {s.yield_years}")
        

if __name__ == "__main__":
    main()
