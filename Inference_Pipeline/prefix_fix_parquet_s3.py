import io
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from urllib.parse import urlparse

SRC_PREFIX = "s3://geoai-demo-data/raw/dataset=era5/state_fips=19/county_fips=ALL/granularity=daily/year=2025/"
DST_PREFIX = "s3://geoai-demo-data/raw_fixed/dataset=era5/state_fips=19/county_fips=ALL/granularity=daily/year=2025/"
DRY_RUN = False  # set True to test without writing

# enforce stable schema
INT32_COLS = ["year", "month", "day"]
STRING_COLS = ["county_fips", "county_name"]

s3 = boto3.client("s3")

def parse_s3(uri: str):
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"Not s3 uri: {uri}")
    return p.netloc, p.path.lstrip("/")

def list_keys(bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

def read_parquet_singlefile_from_s3(bucket: str, key: str) -> pa.Table:
    # Read as a SINGLE parquet file from bytes (no dataset merge)
    resp = s3.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    return pq.read_table(io.BytesIO(body))

def write_parquet_to_s3(bucket: str, key: str, table: pa.Table):
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

def normalize_table(table: pa.Table) -> pa.Table:
    # Convert to pandas for easy dtype enforcement, then back
    df = table.to_pandas()

    # enforce int32 partitions
    for c in INT32_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("int32")

    # enforce strings
    for c in STRING_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # normalize date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # back to arrow table
    return pa.Table.from_pandas(df, preserve_index=False)

if __name__ == "__main__":
    src_bucket, src_prefix_key = parse_s3(SRC_PREFIX)
    dst_bucket, dst_prefix_key = parse_s3(DST_PREFIX)

    parquet_keys = [k for k in list_keys(src_bucket, src_prefix_key) if k.endswith(".parquet")]
    print("Found parquet files:", len(parquet_keys))

    ok, failed = 0, 0

    for k in parquet_keys:
        rel = k[len(src_prefix_key):]
        dst_key = dst_prefix_key.rstrip("/") + "/" + rel

        src_uri = f"s3://{src_bucket}/{k}"
        dst_uri = f"s3://{dst_bucket}/{dst_key}"
        print("\nSRC:", src_uri)
        print("DST:", dst_uri)

        try:
            t = read_parquet_singlefile_from_s3(src_bucket, k)
            t2 = normalize_table(t)

            if not DRY_RUN:
                write_parquet_to_s3(dst_bucket, dst_key, t2)

            ok += 1
        except Exception as e:
            failed += 1
            print("FAIL:", type(e).__name__, str(e))

    print("\nSUMMARY:", {"ok": ok, "failed": failed, "dry_run": DRY_RUN})