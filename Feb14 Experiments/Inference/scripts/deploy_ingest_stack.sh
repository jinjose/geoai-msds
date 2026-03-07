#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# Deploy GeoAI Production Ingestion Stack
# ==========================================================
# Usage:
# bash scripts/deploy_ingest_stack.sh \
#   --region ap-south-1 \
#   --stack geoai-prod-stack \
#   --project geoai \
#   --env prod \
#   --vpc vpc-xxxx \
#   --subnets subnet-aaa,subnet-bbb \
#   --sg sg-xxxx \
#   --ecr <ACCOUNT>.dkr.ecr.ap-south-1.amazonaws.com/geoai-ingestion:latest \
#   --artifact-bucket geoai-artifact-bucket \
#   --gee-secret arn:aws:secretsmanager:...:geoai/gee/service_account_json \
#   --nass-secret arn:aws:secretsmanager:...:geoai/nass/quickstats_api_key
# ==========================================================

REGION=""
STACK=""
PROJECT="geoai"
ENVIRONMENT="prod"
VPC=""
SUBNETS=""
SECURITY_GROUPS=""
ECR_IMAGE=""
ARTIFACT_BUCKET=""
GEE_SECRET=""
NASS_SECRET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --stack) STACK="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --env) ENVIRONMENT="$2"; shift 2 ;;
    --vpc) VPC="$2"; shift 2 ;;
    --subnets) SUBNETS="$2"; shift 2 ;;
    --sg) SECURITY_GROUPS="$2"; shift 2 ;;
    --ecr) ECR_IMAGE="$2"; shift 2 ;;
    --artifact-bucket) ARTIFACT_BUCKET="$2"; shift 2 ;;
    --gee-secret) GEE_SECRET="$2"; shift 2 ;;
    --nass-secret) NASS_SECRET="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ==========================
# Validate Required Params
# ==========================
if [[ -z "$REGION" || -z "$STACK" || -z "$VPC" || -z "$SUBNETS" || -z "$SECURITY_GROUPS" || -z "$ECR_IMAGE" || -z "$ARTIFACT_BUCKET" || -z "$GEE_SECRET" || -z "$NASS_SECRET" ]]; then
  echo ""
  echo "ERROR: Missing required parameters."
  echo ""
  echo "Required:"
  echo "  --region"
  echo "  --stack"
  echo "  --vpc"
  echo "  --subnets"
  echo "  --sg"
  echo "  --ecr"
  echo "  --artifact-bucket"
  echo "  --gee-secret"
  echo "  --nass-secret"
  echo ""
  exit 1
fi

echo "==============================================="
echo "Deploying GeoAI Ingestion Stack"
echo "Region: $REGION"
echo "Stack:  $STACK"
echo "Env:    $ENVIRONMENT"
echo "==============================================="

# ==========================
# Deploy CloudFormation
# ==========================
aws cloudformation deploy \
  --region "$REGION" \
  --stack-name "$STACK" \
  --template-file infra/cloudformation_ingest.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      ProjectName="$PROJECT" \
      Environment="$ENVIRONMENT" \
      VpcId="$VPC" \
      Subnets="$SUBNETS" \
      SecurityGroups="$SECURITY_GROUPS" \
      ECRImage="$ECR_IMAGE" \
      ArtifactBucket="$ARTIFACT_BUCKET" \
      GeeSecretArn="$GEE_SECRET" \
      NassSecretArn="$NASS_SECRET"

echo ""
echo "✅ Stack deployment completed: $STACK"