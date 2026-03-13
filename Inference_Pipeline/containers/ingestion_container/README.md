GeoAI modular ingestion container

Build:
  docker build -t geoai-demo-ingestion:modular .

Run (example):
  docker run --rm     -e AWS_REGION=ap-south-1     -e DATA_BUCKET=geoai-demo-data     -e STATE_FIPS=19     -e COUNTY_FIPS=ALL     -e INGEST_START=2025-01-01     -e INGEST_END=2025-12-31     -e NDVI_GRANULARITY=yearly     -e ERA5_GRANULARITY=yearly     -e STORM_GRANULARITY=daily     -e GEE_SA_SECRET_ID=geoai/gee/service_account_json     -e NASS_API_KEY_SECRET_ID=geoai/nass/quickstats_api_key     -e YIELD_YEARS=2024,2025     geoai-demo-ingestion:modular --dataset all

Shapefile:
  Bundle tl_2023_us_county.{shp,shx,dbf} under data/counties/ in the build context.
  The code will look in /opt/program/data/counties by default.
