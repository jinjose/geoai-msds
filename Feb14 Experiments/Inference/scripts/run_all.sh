#!/usr/bin/env bash
set -euo pipefail

#######################################
# Helpers
#######################################
log()  { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die()  { echo -e "ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# Retry wrapper: retry <retries> <sleep_seconds> -- <command...>
retry() {
  local retries="$1"; shift
  local sleep_s="$1"; shift
  [[ "$1" == "--" ]] || die "retry usage: retry <retries> <sleep_seconds> -- <cmd...>"
  shift

  local n=0
  until "$@"; do
    n=$((n+1))
    if [[ $n -ge $retries ]]; then
      die "Command failed after ${retries} attempts: $*"
    fi
    log "Command failed (attempt $n/${retries}). Retrying in ${sleep_s}s..."
    sleep "$sleep_s"
  done
}

# Ensure S3 bucket exists (idempotent). Works across regions.
ensure_bucket() {
  local bucket="$1"
  local region="$2"
  [[ -z "$bucket" ]] && die "ensure_bucket: bucket name is empty"
  [[ -z "$region" ]] && die "ensure_bucket: region is empty"

  # head-bucket returns 0 if bucket exists and you have access; non-zero otherwise.
  if aws s3api head-bucket --bucket "$bucket" --region "$region" >/dev/null 2>&1; then
    log "S3 bucket exists and is accessible: $bucket"
    return 0
  fi

  log "S3 bucket not found or not accessible, attempting to create: $bucket (region=$region)"

  # S3 bucket names are globally unique. If the name is taken in another account, creation will fail.
  if [[ "$region" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$bucket" --region "$region" >/dev/null
  else
    aws s3api create-bucket --bucket "$bucket" --region "$region" \
      --create-bucket-configuration LocationConstraint="$region" >/dev/null
  fi

  # Apply basic security posture (best effort; ignore if blocked by policy)
  aws s3api put-public-access-block --bucket "$bucket" --region "$region" \
    --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true >/dev/null 2>&1 || true

  aws s3api put-bucket-encryption --bucket "$bucket" --region "$region" \
    --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}' >/dev/null 2>&1 || true

  log "Created bucket (or configured) successfully: $bucket"
}

#######################################
# Config loading
#######################################
CONFIG_FILE="${CONFIG_FILE:-config.env}"

# Allow CLI overrides like: ./scripts/run_all.sh --config config.env --region ap-south-1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2;;
    --region) OVERRIDE_AWS_REGION="$2"; shift 2;;
    --stack)  OVERRIDE_STACK_NAME="$2"; shift 2;;
    --project) OVERRIDE_PROJECT_NAME="$2"; shift 2;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -f "$CONFIG_FILE" ]] || die "Config file not found: $CONFIG_FILE"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

# Apply overrides if provided
AWS_REGION="${OVERRIDE_AWS_REGION:-${AWS_REGION:-}}"
STACK_NAME="${OVERRIDE_STACK_NAME:-${STACK_NAME:-}}"
PROJECT_NAME="${OVERRIDE_PROJECT_NAME:-${PROJECT_NAME:-}}"

#######################################
# Validate config
#######################################
[[ -n "${AWS_REGION:-}" ]] || die "AWS_REGION is required"
[[ -n "${STACK_NAME:-}" ]] || die "STACK_NAME is required"
[[ -n "${PROJECT_NAME:-}" ]] || die "PROJECT_NAME is required"

[[ -n "${MODEL_BUCKET:-}" ]] || die "MODEL_BUCKET is required"
[[ -n "${MODEL_S3_KEY:-}" ]] || die "MODEL_S3_KEY is required"
[[ -n "${LAMBDA_ASSETS_BUCKET:-}" ]] || die "LAMBDA_ASSETS_BUCKET is required"

[[ -n "${ECR_REPO:-}" ]] || die "ECR_REPO is required"
[[ -n "${ECR_TAG:-}" ]] || die "ECR_TAG is required"

[[ -n "${FEATURE_INPUT_DIR:-}" ]] || die "FEATURE_INPUT_DIR is required"

#######################################
# Preflight checks
#######################################
require_cmd aws
require_cmd python
require_cmd bash

log "Preflight: checking AWS identity..."
aws sts get-caller-identity --region "$AWS_REGION" >/dev/null

# Ensure required buckets exist before we proceed (idempotent)
ensure_bucket "$MODEL_BUCKET" "$AWS_REGION"
ensure_bucket "$LAMBDA_ASSETS_BUCKET" "$AWS_REGION"
if [[ -n "${ARTIFACTS_BUCKET:-}" ]]; then
  ensure_bucket "$ARTIFACTS_BUCKET" "$AWS_REGION"
fi


# Docker is required only if building/pushing image
if [[ "${SKIP_BUILD_PUSH_ECR:-false}" != "true" ]]; then
  require_cmd docker
  log "Preflight: checking Docker daemon..."
  docker info >/dev/null 2>&1 || die "Docker daemon is not running. Start Docker Desktop (Windows/Mac) or docker service (Linux)."
fi

#######################################
# Paths
#######################################
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_TAR_LOCAL="${MODEL_TAR_LOCAL:-model.tar.gz}"

#######################################
# Step 1: Package model.tar.gz
#######################################
if [[ "${SKIP_PACKAGE_MODEL:-false}" == "true" ]]; then
  log "SKIP: packaging model.tar.gz"
else
  [[ -n "${MODEL_PKL_PATH:-}" ]] || die "MODEL_PKL_PATH is required (or set SKIP_PACKAGE_MODEL=true)"
  [[ -n "${FEATURE_LIST_PATH:-}" ]] || die "FEATURE_LIST_PATH is required (or set SKIP_PACKAGE_MODEL=true)"
  [[ -f "$MODEL_PKL_PATH" ]] || die "MODEL_PKL_PATH not found: $MODEL_PKL_PATH"
  [[ -f "$FEATURE_LIST_PATH" ]] || die "FEATURE_LIST_PATH not found: $FEATURE_LIST_PATH"

  log "Step 1/5: Packaging SageMaker model artifact: $MODEL_TAR_LOCAL"
  python scripts/package_model_tar.py \
    --model-pkl "$MODEL_PKL_PATH" \
    --feature-list "$FEATURE_LIST_PATH" \
    --out "$MODEL_TAR_LOCAL"
  [[ -f "$MODEL_TAR_LOCAL" ]] || die "Packaging failed: $MODEL_TAR_LOCAL not created"
fi

#######################################
# Step 2: Upload model.tar.gz to S3
#######################################
if [[ "${SKIP_UPLOAD_MODEL:-false}" == "true" ]]; then
  log "SKIP: uploading model.tar.gz to S3"
else
  [[ -f "$MODEL_TAR_LOCAL" ]] || die "model.tar.gz not found locally: $MODEL_TAR_LOCAL"
  log "Step 2/5: Uploading model artifact to S3: s3://$MODEL_BUCKET/$MODEL_S3_KEY"

  retry 3 5 -- aws s3 cp "$MODEL_TAR_LOCAL" "s3://${MODEL_BUCKET}/${MODEL_S3_KEY}" --region "$AWS_REGION"

  log "Verifying model artifact exists in S3..."
  aws s3 ls "s3://${MODEL_BUCKET}/${MODEL_S3_KEY}" --region "$AWS_REGION" >/dev/null \
    || die "Upload verification failed: model not found in s3://${MODEL_BUCKET}/${MODEL_S3_KEY}"
fi

#######################################
# Step 3: Build & push ECR image
#######################################
if [[ "${SKIP_BUILD_PUSH_ECR:-false}" == "true" ]]; then
  log "SKIP: build & push ECR image"
else
  log "Step 3/5: Building & pushing BYOC image to ECR (repo=$ECR_EVAL_REPO tag=$ECR_TAG)"
  retry 2 10 -- bash scripts/build_and_push_processing_ecr.sh \
    --region "$AWS_REGION" \
    --repo "$ECR_REPO" \
    --tag "$ECR_TAG"  
fi

###################################################
# Step 3-1: Build & push processor eval ECR image
##################################################
if [[ "${SKIP_BUILD_PUSH_ECR_EVAL:-false}" == "true" ]]; then
  log "SKIP: build & push processor eval ECR image"
else
  log "Step 3/5: Building & pushing BYOC image to ECR (repo=$ECR_EVAL_REPO tag=$ECR_TAG)"
  retry 2 10 -- bash scripts/build_and_push_processing_ecr.sh \
    --region "$AWS_REGION" \
    --repo "$ECR_EVAL_REPO" \
    --tag "$ECR_TAG"  
fi

#######################################################
# Step 3-2: Build & push processor compare  ECR image
#######################################################
if [[ "${SKIP_BUILD_PUSH_ECR_COMPARE:-false}" == "true" ]]; then
  log "SKIP: build & push processor compare  ECR image"
else
 
  log "Step 3/5: Building & pushing BYOC image to ECR (repo=$ECR_COMPARE_REPO tag=$ECR_TAG)"
  retry 2 10 -- bash scripts/build_and_push_processing_ecr.sh \
    --region "$AWS_REGION" \
    --repo "$ECR_COMPARE_REPO" \
    --tag "$ECR_TAG"
fi

#######################################################
# Step 3-3: Build & push ingest  ECR image
#######################################################
if [[ "${SKIP_BUILD_PUSH_ECR_INGEST:-false}" == "true" ]]; then
  log "SKIP: build & push processor ingest  ECR image"
else
 
  log "Step 3/5: Building & pushing BYOC image to ECR (repo=$ECR_INGEST_REPO tag=$ECR_TAG)"
  retry 2 10 -- bash scripts/build_and_push_ingest_ecr.sh \
    --region "$AWS_REGION" \
    --repo "$ECR_INGEST_REPO" \
    --tag "$ECR_TAG"
fi

#######################################
# Step 4: Deploy CloudFormation
#######################################
if [[ "${SKIP_DEPLOY_STACK:-false}" == "true" ]]; then
  log "SKIP: deploy CloudFormation stack"
else
  log "Step 4/5: Deploying CloudFormation stack: $STACK_NAME"

  DEPLOY_ARGS=(--region "$AWS_REGION" --stack "$STACK_NAME" --project "$PROJECT_NAME" --lambda-assets-bucket "$LAMBDA_ASSETS_BUCKET" --model-bucket "$MODEL_BUCKET")

  if [[ -n "${EXISTING_SAGEMAKER_EXEC_ROLE_ARN:-}" ]]; then
    log "Using existing SageMaker execution role: $EXISTING_SAGEMAKER_EXEC_ROLE_ARN"
    DEPLOY_ARGS+=(--existing-role-arn "$EXISTING_SAGEMAKER_EXEC_ROLE_ARN")
  fi

  retry 2 15 -- bash scripts/deploy_stack.sh "${DEPLOY_ARGS[@]}"
fi

#######################################
# Step 5: Upload batch inputs
#######################################
if [[ "${SKIP_UPLOAD_INPUTS:-false}" == "true" ]]; then
  log "SKIP: upload batch inputs"
else
  [[ -d "$FEATURE_INPUT_DIR" ]] || die "FEATURE_INPUT_DIR not found: $FEATURE_INPUT_DIR"

  # Determine artifacts bucket: use explicit config if set, otherwise try stack output
  if [[ -z "${ARTIFACTS_BUCKET:-}" ]]; then
    log "ARTIFACTS_BUCKET not set. Attempting to read stack output..."
    ARTIFACTS_BUCKET="$(aws cloudformation describe-stacks \
      --region "$AWS_REGION" \
      --stack-name "$STACK_NAME" \
      --query "Stacks[0].Outputs[?OutputKey=='ArtifactsBucket'].OutputValue | [0]" \
      --output text 2>/dev/null || true)"
  fi

  [[ -n "${ARTIFACTS_BUCKET:-}" && "${ARTIFACTS_BUCKET}" != "None" ]] || die "Could not determine ARTIFACTS_BUCKET. Set ARTIFACTS_BUCKET in config.env."

  log "Step 5/5: Uploading batch input features to s3://$ARTIFACTS_BUCKET/batch/input/"
  retry 3 5 -- bash scripts/upload_batch_inputs.sh \
    --region "$AWS_REGION" \
    --bucket "$ARTIFACTS_BUCKET" \
    --input-dir "$FEATURE_INPUT_DIR"
  
  retry 3 5 -- bash scripts/upload_cutoff_inputs.sh \
    "$AWS_REGION" \
    "$ARTIFACTS_BUCKET" \
    "$(date +%Y-%m-%d)" \
    "D:/MS_DataScience/MSDS_498_Capstone_Final/GIT/geoai-msds/Feb14 Experiments/Inference/data/features_frozen/out"
fi

log "✅ All steps completed successfully."
log "Next: wait for EventBridge schedule OR invoke the scheduler Lambda manually to trigger immediately."
