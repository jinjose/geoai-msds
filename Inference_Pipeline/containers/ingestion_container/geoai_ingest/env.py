import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    aws_region: str
    data_bucket: str
    raw_prefix: str
    state_fips: str
    county_fips: str
    ingest_start: str
    ingest_end: str
    ndvi_granularity: str
    era5_granularity: str
    storm_granularity: str
    gee_sa_secret_id: str
    nass_api_key_secret_id: str
    yield_years: str

def _get_region() -> str:
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1"

def load_settings() -> Settings:
    # SageMaker Processing may omit env vars if the value resolves to null.
    # Use getenv() with safe defaults for optional values to avoid KeyError.
    ingest_start = os.getenv("INGEST_START", "")
    ingest_end = os.getenv("INGEST_END", "")

    return Settings(
        aws_region=_get_region(),
        data_bucket=os.environ["DATA_BUCKET"],
        raw_prefix=os.environ.get("RAW_PREFIX", "raw"),

        state_fips=str(os.environ.get("STATE_FIPS", "0")).zfill(2),
        county_fips=("ALL" if os.environ.get("COUNTY_FIPS", "ALL").upper() == "ALL"
                     else str(os.environ["COUNTY_FIPS"]).zfill(3)),

        ingest_start=ingest_start.strip(),
        ingest_end=ingest_end.strip(),

        ndvi_granularity=os.environ.get("NDVI_GRANULARITY", "daily"),
        era5_granularity=os.environ.get("ERA5_GRANULARITY", "daily"),
        storm_granularity=os.environ.get("STORM_GRANULARITY", "daily"),

        gee_sa_secret_id=os.environ["GEE_SA_SECRET_ID"],
        nass_api_key_secret_id=os.environ["NASS_API_KEY_SECRET_ID"],

        yield_years=os.environ.get("YIELD_YEARS", ""),
    )
