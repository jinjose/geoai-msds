
# GeoAI Inference Pipeline

This directory contains a modular, production-grade, end-to-end inference pipeline for crop yield prediction using AWS SageMaker Batch Transform. The pipeline is designed for scalable, cost-effective, and reproducible inference and evaluation across multiple geographies and model versions.

---

## Pipeline Overview

The pipeline consists of four main containerized stages:

1. **Ingestion**: Downloads and prepares raw data (NDVI, weather, storm, yield) from external sources and S3, partitioned by state/county/year.
2. **Feature Engineering**: Processes raw data into model-ready, seasonally-frozen features (no label leakage), outputting Parquet files for each geography/season.
3. **Inference**: Runs MLflow models (or native baselines) on frozen features using SageMaker Batch Transform, producing predictions per county/year/season.
4. **Evaluation**: Joins predictions with actuals, computes metrics (MAE, RMSE, R2, etc.), and writes results to S3 for analysis and reporting.

All steps are orchestrated via AWS Step Functions, with infrastructure-as-code provided (see infra/ and scripts/).

---

## Folder Structure

- `containers/ingestion_container/`   – Raw data ingestion (Google Earth Engine, USDA, ERA5, etc.)
- `containers/feature_container/`     – Feature engineering and freezing (no label leakage)
- `containers/inference_container/`   – Model inference (MLflow or native)
- `containers/evaluation_container/`  – Evaluation and metrics computation
- `scripts/`                         – Build, deploy, and utility scripts
- `exported_models/`                 – Model artifacts for inference
- `infra/`                           – CloudFormation templates and pipeline configs
- `utilities/`                       – Data utilities and helpers

---

## Prerequisites

- AWS CLI v2 (configured for your account/region)
- Docker
- Python 3.10+
- Access to required AWS resources (S3, SageMaker, Step Functions, SecretsManager)

---

## Quick Start

### 1. Build & Push Docker Images

```sh
chmod +x scripts/*.sh
./scripts/deploy_all.sh
```

### 2. Update Model Registry

Upload your model folders to S3 and update the registry JSON:

```
s3://<DATA_BUCKET>/config/model_registry.json
```

### 3. Deploy Infrastructure

Deploy the CloudFormation stack using the provided template in `infra/`.

### 4. Run the Pipeline

Start a Step Functions execution (example):

```sh
aws stepfunctions start-execution --state-machine-arn <ARN> --input '{
  "run_date": "2026-02-22",
  "state_fips": "19",
  "county_fips": "001",
  "feature_seasons": ["jun01", "jul01", "jul15", "aug01"],
  "backfill_start_year": 2014,
  "secrets": {
    "gee_sa_secret_id": "<SecretsManagerIdForGEE-SA>",
    "nass_api_key_secret_id": "<SecretsManagerIdForQuickStatsKey>"
  }
}'
```

---

## Container Details

### Ingestion Container
- Downloads and harmonizes raw data (NDVI, ERA5, storm, yield) for specified geographies and years.
- Supports Google Earth Engine and USDA QuickStats APIs (credentials via AWS SecretsManager).
- Outputs partitioned Parquet files to S3 under `raw/`.

### Feature Container
- Reads raw Parquet data, applies cleaning, smoothing, and feature engineering logic.
- Freezes features for each season (e.g., jun01, jul01, etc.) with no label leakage.
- Outputs `features_frozen/` Parquet files (no yield labels) for inference.

### Inference Container
- Loads MLflow models or native baselines.
- Accepts frozen features as input (Parquet or CSV).
- Outputs predictions per county/year/season to S3 under `predictions/`.

### Evaluation Container
- Joins predictions with actuals.
- Computes metrics (MAE, RMSE, MAPE, R2, Bias, N) for each model/season.
- Outputs metrics and comparison files to S3 under `evaluation/`.

---

## Data Flow

1. **Ingestion**: S3 `raw/dataset=.../state_fips=.../county_fips=.../year=.../`
2. **Feature Freezing**: S3 `features_frozen/state_fips=.../county_fips=.../feature_season=.../run_date=.../`
3. **Inference**: S3 `predictions/state_fips=.../county_fips=.../predict_year=.../feature_season=.../run_date=.../model=.../`
4. **Evaluation**: S3 `evaluation/run_date=.../state_fips=.../county_fips=.../predict_year=.../`

---

## Best Practices & Notes

- All data is stored in Parquet format for efficiency and scalability.
- Feature freezing ensures no label leakage; yield labels are excluded from inference inputs.
- Model registry supports multiple models and versions (MLflow artifacts or native code).
- Evaluation is fully decoupled and can be rerun as new actuals become available.
- All containers are designed for AWS SageMaker Processing/Batch Transform, but can be run locally for testing.

---

## Customization

- Plug in your own ingestion logic in `ingestion_container/` (see `ingest_raw.py`).
- Add or modify feature engineering in `feature_container/app/build_new_features.py`.
- Register new models in the model registry and place artifacts in `exported_models/`.
- Extend evaluation logic in `evaluation_container/evaluate.py` as needed.

---

## References

- See each container's README or main script for environment variables and advanced usage.
- For troubleshooting, see logs in CloudWatch and outputs in S3.

---

**Contact:** For questions or contributions, please contact the project maintainers.
