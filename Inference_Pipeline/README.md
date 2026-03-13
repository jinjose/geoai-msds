# GeoAI Crop Yield – SageMaker Batch Transform Demo Pipeline (Capstone)

This repo deploys a cost-effective, SageMaker-first inference pipeline that supports:
- backfill default from 2014 for new geographies (state/county)
- incremental ingestion for previously ingested geographies (tracked in DynamoDB)
- frozen features for 4 cutoff seasons: jun01, jul01, jul15, aug01
- model-agnostic evaluation grid: 4 feature inputs × N MLflow models (your current N=4 => 16 jobs)
- SageMaker Batch Transform (no always-on endpoints)
- Separate evaluation stage (no data leakage). Frozen feature CSVs MUST NOT include yield_bu_acre.

## What you need installed
- AWS CLI v2 configured (ap-south-1)
- Docker
- Python 3.10+

## Quick start (demo)
1) Build & push images to ECR, then deploy CloudFormation:

   chmod +x scripts/*.sh
   ./scripts/deploy_all.sh

2) Update model registry (S3) with your model folders:
   s3://<DATA_BUCKET>/config/model_registry.json

3) Start a pipeline execution (example):
   aws stepfunctions start-execution --state-machine-arn <ARN> --input '{
     "run_date":"2026-02-22",
     "state_fips":"19",
     "county_fips":"001",
     "feature_seasons":["jun01","jul01","jul15","aug01"],
     "backfill_start_year":2014,
     "secrets":{
       "gee_sa_secret_id":"<SecretsManagerIdForGEE-SA>",
       "nass_api_key_secret_id":"<SecretsManagerIdForQuickStatsKey>"
     }
   }'

## Important notes
- This is an infrastructure + runnable skeleton. You will plug your final business logic into:
  - containers/ingestion_container (uses your ingest_raw.py; expects Google Earth Engine secret, etc.)
  - containers/feature_container (uses your feature extraction code; writes inference-ready features.csv WITHOUT yield_bu_acre)
  - containers/evaluation_container (joins predictions with actual yield and computes metrics)
- Batch Transform input location used by state machine:
  s3://<DATA_BUCKET>/features_frozen/state_fips=.../county_fips=.../feature_season=.../run_date=.../features.csv


## Professional production-style choices
- Parquet-only lake under s3://<DATA_BUCKET>/raw/ (partitioned)
- Frozen features stored as Parquet under s3://<DATA_BUCKET>/features/
- Predictions stored as Parquet under s3://<DATA_BUCKET>/predictions/
- Baseline model (lag1_baseline) runs natively using lag1_yield feature; other models load MLflow artifacts.
