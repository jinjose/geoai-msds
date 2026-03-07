# GeoAI – LightGBM BYOC Batch Transform on SageMaker (Hybrid)

This repo packages a **LightGBM Booster** into `model.tar.gz`, builds a **BYOC** inference container, and runs **daily Batch Transform** via **EventBridge → Lambda → SageMaker**.

## What runs where

- **Local (dev/lab):** feature engineering + training + MLflow tracking/registry
- **AWS (ops):**
  - **S3**: inputs (`batch/input/`), outputs (`batch/output/`), model artifact (`geoai/model.tar.gz`)
  - **ECR**: BYOC inference image
  - **SageMaker Model**: points to ECR image + S3 model.tar.gz
  - **Lambda**: starts a Batch Transform job
  - **EventBridge**: triggers Lambda once per day

## Quick start (AWS)

### 0) Prereqs
- AWS CLI configured (`aws sts get-caller-identity` works)
- Docker Desktop running (Windows users: Docker Desktop **must** be started)
- An S3 bucket to store CloudFormation lambda asset zip (can be the same as the artifacts bucket, but easiest is any bucket you own)

### 1) Package `model.tar.gz`
Create a SageMaker model artifact locally:

```bash
python scripts/package_model_tar.py --model-pkl model_artifacts/model.pkl --feature-list model_artifacts/feature_list.txt --out model.tar.gz
```

Upload it to **S3** under the key `geoai/model.tar.gz` **before** deploying the stack (the stack has a preflight check):

```bash
aws s3 cp model.tar.gz s3://<YOUR_BUCKET>/geoai/model.tar.gz --region ap-south-1
```

### 2) Build & push the BYOC image to ECR
```bash
bash scripts/build_and_push_ecr.sh --region ap-south-1 --repo geoai-lgbm-group6-inference --tag v1
```

> If you see `dockerDesktopLinuxEngine/_ping` errors on Windows, Docker Desktop isn’t running or the Linux engine isn’t available.

### 3) Deploy CloudFormation
```bash
bash scripts/deploy_stack.sh --region ap-south-1 --stack geoai-lgbm-group6 --project geoai --lambda-assets-bucket <YOUR_BUCKET>
```

If your AWS account can’t create IAM roles, pass an existing SageMaker execution role:

```bash
bash scripts/deploy_stack.sh --region ap-south-1 --stack geoai-lgbm-group6 --project geoai   --lambda-assets-bucket <YOUR_BUCKET>   --existing-role-arn arn:aws:iam::<acct>:role/<existing-sagemaker-role>
```

### 4) Upload batch inputs (features)
Put your features CSVs in `data/features/` (or point to your folder) and upload:

```bash
bash scripts/upload_batch_inputs.sh --region ap-south-1 --bucket <ARTIFACTS_BUCKET_FROM_STACK_OUTPUT> --input-dir data/features
```

### 5) Verify outputs
After the daily run, predictions are written to:

`s3://<artifacts-bucket>/batch/output/predictions.csv`

## Notes / Common issues

### ECR push “not authorized”
Your IAM user needs ECR permissions such as `ecr:InitiateLayerUpload`, `ecr:PutImage`, etc. If you don’t have admin, ask for an IAM policy update.

### CloudFormation fails to create IAM roles
If your user lacks `iam:CreateRole` (common in student accounts), deploy using `--existing-role-arn`.

### SageMaker instance quotas are 0
If Batch Transform instance quotas are 0, request quota increases via Service Quotas / AWS Support (you already did this once).

## Repository layout

- `src/geoai/` – shared python package (inference utilities, logging)
- `docker/` – BYOC Batch Transform container (`Dockerfile`, `batch_runner.py`)
- `lambda/` – scheduler Lambda that starts Transform jobs
- `infra/` – CloudFormation template (S3, ECR, IAM, Lambda, EventBridge, SageMaker Model)
- `scripts/` – build/deploy helpers



##### How to deloy the Zip for the git bash 
Here's another, slightly different, set of instructions to install zip for git bash on windows:

Navigate to this sourceforge page (https://sourceforge.net/projects/gnuwin32/files/zip/3.0/)
Download zip-3.0-bin.zip
In the zipped file, in the bin folder, find the file zip.exe.
Extract the file zip.exe to your mingw64 bin folder (for me: C:\Program Files\Git\mingw64\bin)
Navigate to to this sourceforge page (https://sourceforge.net/projects/gnuwin32/files/bzip2/1.0.5/)
Download bzip2-1.0.5-bin.zip
In the zipped file, in the bin folder, find the file bzip2.dll
Extract bzip2.dll to your mingw64\bin folder (same folder as above: C:\Program Files\Git\mingw64\bin)