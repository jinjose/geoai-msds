import json
import ee
from .secrets import secrets_json

def init_gee(sa_secret_id: str, region: str) -> None:
    """Initialize Earth Engine using a service-account JSON stored in Secrets Manager."""
    sa_info = secrets_json(sa_secret_id, region)

    sa_path = "gee_service_account.json"
    with open(sa_path, "w", encoding="utf-8") as f:
        json.dump(sa_info, f)

    creds = ee.ServiceAccountCredentials(sa_info["client_email"], sa_path)
    ee.Initialize(creds, project=sa_info.get("project_id"))
