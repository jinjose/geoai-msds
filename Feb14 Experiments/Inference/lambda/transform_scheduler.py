"""Lambda to start a SageMaker Batch Transform job on a schedule.

Env vars:
  MODEL_NAME: SageMaker Model name
  SAGEMAKER_ROLE_ARN: role to pass (execution role)
  INPUT_S3_PREFIX: s3://bucket/prefix/  (must end with /)
  OUTPUT_S3_PREFIX: s3://bucket/prefix/
  INSTANCE_TYPE: e.g., ml.m5.large
  INSTANCE_COUNT: "1"
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3

sm = boto3.client("sagemaker")
s3 = boto3.client("s3")

def _parse_s3(s3_uri: str):
    u = urlparse(s3_uri)
    if u.scheme != "s3":
        raise ValueError(f"Expected s3://..., got {s3_uri}")
    return u.netloc, u.path.lstrip("/")

def _has_csv(s3_uri: str) -> bool:
    bucket, key = _parse_s3(s3_uri)
    # if user passed a file path, check that exact key
    if key.lower().endswith(".csv"):
        s3.head_object(Bucket=bucket, Key=key)
        return True

    # otherwise treat as prefix
    prefix = key.rstrip("/") + "/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=50)
    return any(o["Key"].lower().endswith(".csv") for o in resp.get("Contents", []))


def handler(event, context):
    model_name = os.environ["MODEL_NAME"]
    input_s3 = os.environ["INPUT_S3_PREFIX"]
    output_s3 = os.environ["OUTPUT_S3_PREFIX"]
    instance_type = os.environ.get("INSTANCE_TYPE", "ml.m5.large")
    instance_count = int(os.environ.get("INSTANCE_COUNT", "1"))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job_name = f"geoai-bt-{ts}-{uuid.uuid4().hex[:8]}"
    if not _has_csv(input_s3):
        raise RuntimeError(f"No .csv found under INPUT_S3_PREFIX={input_s3}. "
                           f"Set it to s3://geoai-model-bucket/batch/input/ or the exact csv file.")
                           
    resp = sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        MaxConcurrentTransforms=1,
        MaxPayloadInMB=50,
        BatchStrategy="MultiRecord",
        TransformOutput={
            "S3OutputPath": output_s3.rstrip("/"),
            "AssembleWith": "Line",
            "Accept": "text/csv",
        },
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_s3.rstrip("/")}},
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        TransformResources={
            "InstanceType": instance_type,
            "InstanceCount": instance_count,
        },
    )

    return {"job_name": job_name, "response": resp}
