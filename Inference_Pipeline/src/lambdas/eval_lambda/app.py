import json
import os
import boto3
from datetime import datetime, timezone

s3 = boto3.client("s3")
DATA_BUCKET = os.environ["DATA_BUCKET"]

def lambda_handler(event, context):
    # Write a small summary pointer file (the Processing evaluation writes metrics artifacts)
    if "run_date" not in event:
        raise ValueError("run_date missing from event")
    
    run_date = event.get("run_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")    
    key = f"evaluation/run_date={run_date}/_SUMMARY.json"
    s3.put_object(
        Bucket=DATA_BUCKET,
        Key=key,
        Body=json.dumps({"ok": True, "run_date": run_date, "event": event})[:200000].encode("utf-8"),
        ContentType="application/json",
    )
    return {"status": "ok", "summary": f"s3://{DATA_BUCKET}/{key}"}
