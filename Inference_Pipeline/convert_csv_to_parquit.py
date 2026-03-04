import boto3
import pandas as pd
from urllib.parse import urlparse

# =========================
# CONFIG
# =========================
S3_PREFIX = "s3://geoai-demo-data/raw/dataset=yield/state_fips=19/county_fips=ALL/granularity=yearly/"   # <-- change me
DRY_RUN = False                           # True = no writes/deletes
DELETE_ORIGINAL_CSV = True              # True = delete CSV after successful parquet write
OVERWRITE_PARQUET = False                # True = overwrite existing .parquet
WRITE_TO_NEW_PREFIX = False              # True = write parquet under a different prefix
NEW_PREFIX = "s3://geoai-demo-data/parquet/"  # used only if WRITE_TO_NEW_PREFIX=True

# Read options (tune if needed)
READ_CSV_KWARGS = {
    # "dtype": {"year": "Int64"},  # example
    "low_memory": False
}

# Parquet options
PARQUET_ENGINE = "pyarrow"


# =========================
# HELPERS
# =========================
def parse_s3_uri(s3_uri: str):
    p = urlparse(s3_uri)
    if p.scheme != "s3":
        raise ValueError(f"Not an s3 uri: {s3_uri}")
    bucket = p.netloc
    key = p.path.lstrip("/")
    return bucket, key


def s3_join(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def list_keys_recursive(bucket: str, prefix: str):
    """
    Recursively lists all object keys under prefix using pagination.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def key_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def map_to_new_prefix(src_bucket, src_key, src_prefix_key, dst_prefix_uri):
    """
    Keep relative path under src_prefix, but write under dst_prefix.
    Example:
      src_prefix: raw/dataset=ndvi/
      src_key:    raw/dataset=ndvi/year=2025/part.csv
      dst_prefix: parquet/
      =>          parquet/dataset=ndvi/year=2025/part.parquet
    """
    dst_bucket, dst_prefix_key = parse_s3_uri(dst_prefix_uri)
    rel = src_key[len(src_prefix_key):] if src_key.startswith(src_prefix_key) else src_key
    return dst_bucket, (dst_prefix_key.rstrip("/") + "/" + rel.lstrip("/"))


def delete_s3_objects(bucket: str, keys):
    """
    Deletes in batches of 1000.
    """
    s3 = boto3.client("s3")
    keys = list(keys)
    for i in range(0, len(keys), 1000):
        batch = keys[i:i+1000]
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch]}
        )


def convert_one_csv(bucket, csv_key, src_prefix_key):
    csv_uri = s3_join(bucket, csv_key)

    # Destination key
    parquet_key = csv_key[:-4] + ".parquet"  # replace .csv
    dst_bucket, dst_key = bucket, parquet_key

    if WRITE_TO_NEW_PREFIX:
        dst_bucket, dst_key = map_to_new_prefix(bucket, parquet_key, src_prefix_key, NEW_PREFIX)

    parquet_uri = s3_join(dst_bucket, dst_key)

    # Skip if parquet exists and not overwriting
    if (not OVERWRITE_PARQUET) and key_exists(dst_bucket, dst_key):
        print(f"SKIP (parquet exists): {parquet_uri}")
        return "skipped_exists"

    print(f"READ : {csv_uri}")
    print(f"WRITE: {parquet_uri}")

    if DRY_RUN:
        return "dry_run"

    # Read CSV from S3 and write Parquet to S3
    df = pd.read_csv(csv_uri, **READ_CSV_KWARGS)

    # (Optional) minimal normalization example:
    # if "year" in df.columns:
    #     df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df.to_parquet(parquet_uri, index=False, engine=PARQUET_ENGINE)

    # Delete CSV only after successful parquet write
    if DELETE_ORIGINAL_CSV:
        boto3.client("s3").delete_object(Bucket=bucket, Key=csv_key)
        print(f"DELETE: {csv_uri}")

    return "converted"


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    bucket, prefix_key = parse_s3_uri(S3_PREFIX)

    print("===================================================")
    print(f"Scanning recursively under: s3://{bucket}/{prefix_key}")
    print(f"DRY_RUN={DRY_RUN} | DELETE_ORIGINAL_CSV={DELETE_ORIGINAL_CSV} | "
          f"OVERWRITE_PARQUET={OVERWRITE_PARQUET} | WRITE_TO_NEW_PREFIX={WRITE_TO_NEW_PREFIX}")
    if WRITE_TO_NEW_PREFIX:
        print(f"NEW_PREFIX={NEW_PREFIX}")
    print("===================================================\n")

    # Collect CSV keys
    csv_keys = [k for k in list_keys_recursive(bucket, prefix_key) if k.endswith(".csv")]

    print(f"Found {len(csv_keys)} CSV files.\n")
    if not csv_keys:
        print("Nothing to do.")
        raise SystemExit(0)

    stats = {"converted": 0, "skipped_exists": 0, "dry_run": 0, "failed": 0}

    for csv_key in csv_keys:
        try:
            res = convert_one_csv(bucket, csv_key, prefix_key)
            stats[res] = stats.get(res, 0) + 1
        except Exception as e:
            stats["failed"] += 1
            print(f"FAIL: s3://{bucket}/{csv_key}")
            print(f"  -> {type(e).__name__}: {e}\n")

    print("\n================ SUMMARY ================")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("========================================\n")