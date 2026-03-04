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

def load_settings() -> Settings:
    return Settings(
        aws_region=os.environ["AWS_REGION"],
        data_bucket=os.environ["DATA_BUCKET"],
        raw_prefix=os.environ.get("RAW_PREFIX", "raw"),

        state_fips=str(os.environ["STATE_FIPS"]).zfill(2),
        county_fips=("ALL" if os.environ.get("COUNTY_FIPS", "ALL").upper() == "ALL"
                     else str(os.environ["COUNTY_FIPS"]).zfill(3)),

        ingest_start=os.environ["INGEST_START"],
        ingest_end=os.environ["INGEST_END"],

        ndvi_granularity=os.environ.get("NDVI_GRANULARITY", "daily"),
        era5_granularity=os.environ.get("ERA5_GRANULARITY", "daily"),
        storm_granularity=os.environ.get("STORM_GRANULARITY", "daily"),

        gee_sa_secret_id=os.environ["GEE_SA_SECRET_ID"],
        nass_api_key_secret_id=os.environ["NASS_API_KEY_SECRET_ID"],

        yield_years=os.environ.get("YIELD_YEARS", ""),
    )
