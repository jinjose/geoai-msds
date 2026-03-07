import boto3
from urllib.parse import urlparse

# =====================================
# CONFIG
# =====================================
S3_PREFIX = "s3://geoai-demo-data/raw/dataset=era5/state_fips=19/county_fips=ALL/granularity=daily/year=2025/"
DRY_RUN = False   # IMPORTANT: Set False to actually delete


# =====================================
# HELPERS
# =====================================

def parse_s3_path(s3_path):
    parsed = urlparse(s3_path)
    return parsed.netloc, parsed.path.lstrip("/")


def list_objects(bucket, prefix):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    return keys


def delete_keys(bucket, keys):
    s3 = boto3.client("s3")

    # S3 delete supports max 1000 per call
    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        delete_payload = {"Objects": [{"Key": k} for k in batch]}
        s3.delete_objects(Bucket=bucket, Delete=delete_payload)


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    bucket, prefix = parse_s3_path(S3_PREFIX)

    print(f"\nScanning bucket: {bucket}")
    print(f"Prefix: {prefix}")

    all_keys = list_objects(bucket, prefix)

    csv_keys = [k for k in all_keys if k.endswith(".csv")]

    print(f"\nFound {len(csv_keys)} CSV files")

    if not csv_keys:
        print("No CSV files to delete.")
        exit()

    for key in csv_keys[:20]:
        print("  ", key)

    if DRY_RUN:
        print("\n⚠ DRY RUN ENABLED — No files deleted.")
        print("Set DRY_RUN = False to perform actual deletion.")
    else:
        delete_keys(bucket, csv_keys)
        print(f"\n✅ Deleted {len(csv_keys)} CSV files.")

    print("\n🎯 Done.")