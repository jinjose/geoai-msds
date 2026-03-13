import os
import boto3
from datetime import datetime

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.environ["REGISTRY_TABLE"])

def get_last_ingested(dataset_name: str):
    resp = table.get_item(Key={"pk": dataset_name})
    if "Item" not in resp:
        return None
    return resp["Item"].get("last_ingested_date")

def update_last_ingested(dataset_name: str, last_date: str):
    table.put_item(
        Item={
            "pk": dataset_name,
            "last_ingested_date": last_date,
            "status": "OK",
            "updated_at": datetime.utcnow().isoformat()
        }
    )