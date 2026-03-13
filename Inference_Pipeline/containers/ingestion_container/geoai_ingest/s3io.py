import io
from dataclasses import dataclass
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

def s3_upload(region: str, bucket: str, key: str, body: bytes, content_type: str | None = None):
    s3 = boto3.client("s3", region_name=region)
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    s3.put_object(Bucket=bucket, Key=key, Body=body, **extra)

def partition_prefix(base: S3Base, dataset: str, state_fips: str, county_fips: str, gran: str,
                     year: int, month: int | None = None, day: int | None = None, hour: int | None = None) -> str:
    parts = [
        base.prefix.rstrip("/"),
        f"dataset={dataset}",
        f"state_fips={state_fips}",
        f"county_fips={county_fips}",
        f"granularity={gran}",
        f"year={year:04d}",
    ]
    if month is not None:
        parts.append(f"month={month:02d}")
    if day is not None:
        parts.append(f"day={day:02d}")
    if hour is not None:
        parts.append(f"hour={hour:02d}")
    return "/".join(parts) + "/"

def write_csv_parquet(aws_region, out, key_prefix, df: pd.DataFrame, basename="part"):
    """
    Permanent fix:
    - Writes ONLY parquet (no csv)
    - Enforces stable schema for common partition columns
    - Avoids future month int32/int64 drift across files
    """
    if df is None or len(df) == 0:
        # still create no file; feature reader should handle missing partitions
        return

    df = df.copy()

    # ---- stable partition columns (critical) ----
    for c in ["year", "month", "day"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("int32")

    # date format stability (optional but recommended)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # avoid pandas nullable dtypes leaking into parquet inconsistently
    # (convert all integer extension types to int32 if safe)
    for c in df.columns:
        if str(df[c].dtype) == "Int64":
            # choose int32 for stability (or int64 if you prefer)
            df[c] = df[c].astype("float64")  # allow NaN
            # if you never have NaN, you can do: df[c] = df[c].astype("int32")

    # ---- write parquet to S3 via out.put_bytes (or equivalent) ----
    table = pa.Table.from_pandas(df, preserve_index=False)

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")

    parquet_key = f"{key_prefix.rstrip('/')}/{basename}.parquet"
    print(f"Writing parquet to s3://{out.bucket}/{parquet_key} ({len(buf.getvalue())} bytes)")
    s3_upload(
        aws_region,
        out.bucket,
        parquet_key,
        buf.getvalue(),
        content_type="application/octet-stream",
    )
    print(f"Wrote parquet to s3://{out.bucket}/{parquet_key} ({len(buf.getvalue())} bytes)")

    
# def write_csv_parquet(region: str, base: S3Base, key_prefix: str, df: pd.DataFrame, basename: str = "part"):
#     # csv_buf = io.StringIO()
#     # df.to_csv(csv_buf, index=False)
#     # s3_upload(region, base.bucket, f"{key_prefix}{basename}.csv", csv_buf.getvalue().encode("utf-8"), "text/csv")

#     table = pa.Table.from_pandas(df)
#     pq_buf = io.BytesIO()
#     pq.write_table(table, pq_buf)
#     s3_upload(region, base.bucket, f"{key_prefix}{basename}.parquet", pq_buf.getvalue(), "application/octet-stream")
