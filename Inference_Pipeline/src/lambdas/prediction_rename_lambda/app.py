import boto3

s3 = boto3.client("s3")

def _parse_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    path = uri[5:]
    bucket, key = path.split("/", 1)
    return bucket, key

def _head(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def lambda_handler(event, context):
    source_s3_uri = event["source_s3_uri"]
    dest_s3_uri = event["dest_s3_uri"]

    src_bucket, src_key = _parse_s3_uri(source_s3_uri)
    dst_bucket, dst_key = _parse_s3_uri(dest_s3_uri)

    # If destination already exists, do nothing
    if _head(dst_bucket, dst_key):
        return {
            "status": "exists",
            "source_s3_uri": source_s3_uri,
            "dest_s3_uri": dest_s3_uri
        }

    # If source does not exist, do nothing but succeed
    if not _head(src_bucket, src_key):
        return {
            "status": "missing_source",
            "source_s3_uri": source_s3_uri,
            "dest_s3_uri": dest_s3_uri
        }

    s3.copy_object(
        Bucket=dst_bucket,
        Key=dst_key,
        CopySource={"Bucket": src_bucket, "Key": src_key}
    )

    return {
        "status": "copied",
        "source_s3_uri": source_s3_uri,
        "dest_s3_uri": dest_s3_uri
    }