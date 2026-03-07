#!/usr/bin/env bash
set -euo pipefail

# One-command deploy (after you uploaded model.tar.gz).
# Usage:
#   bash scripts/deploy_stack.sh --region ap-south-1 --stack geoai-lgbm-group6 --project geoai --lambda-assets-bucket <bucket>

REGION=""
STACK=""
PROJECT="geoai"
LAMBDA_ASSETS_BUCKET=""
EXISTING_ROLE_ARN=""
MODEL_BUCKET="geoai-model-bucket"
ECR_EVAL_REPO="geoai-lgbm-group6-evalprocess"
ECR_COMPARE_REPO="geoai-lgbm-group6-compareprocess"
ECR_IMAGE="geoai-lgbm-group6-ingestprocess"
ECR_TAG="v1"
ENVIRONMENT="prod"
#aws ec2 describe-vpcs --filters Name=isDefault,Values=true
VPC="vpc-0e8d8ddb519a5e32f" 
#aws ec2 describe-subnets --filters Name=default-for-az,Values=true --query "Subnets[*].SubnetId"
SUBNETS="subnet-078106eac0533e025,subnet-00f533532ffc9b18e,subnet-0c92a2688b990bb80,subnet-00ce694a5b438318e"
#aws ec2 describe-security-groups --filters Name=group-name,Values=default --query "SecurityGroups[*].GroupId"
SECURITY_GROUPS="sg-05e324d86c8dbd184" 
ARTIFACT_BUCKET="geoai-artifact-bucket"
GEE_SECRET_ARN="arn:aws:secretsmanager:ap-south-1:172023108179:secret:geoai/gee/service_account_json-MEHEnL"
NASS_SECRET_ARN="arn:aws:secretsmanager:ap-south-1:172023108179:secret:geoai/nass/quickstats_api_key-DViVxU"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --stack) STACK="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --lambda-assets-bucket) LAMBDA_ASSETS_BUCKET="$2"; shift 2 ;;
    --model-bucket) MODEL_BUCKET="$2"; shift 2 ;;
    --existing-role-arn) EXISTING_ROLE_ARN="$2"; shift 2 ;;
    *) echo "Unknown arg $1"; exit 1 ;;
  esac
done

if [[ -z "${REGION}" || -z "${STACK}" || -z "${LAMBDA_ASSETS_BUCKET}" ]]; then
  echo "Usage: --region <region> --stack <name> --lambda-assets-bucket <bucket> [--existing-role-arn <arn>]"
  exit 1
fi

# Package lambda
bash scripts/package_lambda.sh --region "${REGION}" --bucket "${LAMBDA_ASSETS_BUCKET}" --key "lambda/transform_scheduler.zip"

PARAMS=(
  ParameterKey=ProjectName,ParameterValue="${PROJECT}"
  ParameterKey=LambdaCodeS3Bucket,ParameterValue="${LAMBDA_ASSETS_BUCKET}"
  ParameterKey=LambdaCodeS3Key,ParameterValue="lambda/transform_scheduler.zip"
  ParameterKey=ContainerImageTag,ParameterValue="v1"
  ParameterKey=ArtifactsBucketName,ParameterValue="${MODEL_BUCKET}"
  ParameterKey=EvalProcessingImageUri,ParameterValue="${ECR_EVAL_REPO}:${ECR_TAG}"
  ParameterKey=SageMakerModel,ParameterValue="${PROJECT}-model"
  ParameterKey=CutoffStateMachineName,ParameterValue="${PROJECT}-cutoff-state-machine"
  ParameterKey=CutoffStateMachine,ParameterValue="${PROJECT}-cutoff-state-machine"
  ParameterKey=CutoffStateMachineRole,ParameterValue="${PROJECT}-cutoff-state-machine-role"
  ParameterKey=CompareProcessingImageUri,ParameterValue="${ECR_COMPARE_REPO}:${ECR_TAG}"
  ParameterKey=Environment,ParameterValue="${ENVIRONMENT}" 
  ParameterKey=VpcId,ParameterValue="$VPC" 
  ParameterKey=Subnets,ParameterValue="$SUBNETS" 
  ParameterKey=SecurityGroups,ParameterValue="$SECURITY_GROUPS" 
  ParameterKey=ECRImage,ParameterValue="$ECR_IMAGE"
  ParameterKey=ArtifactBucket,ParameterValue="$ARTIFACT_BUCKET" 
  ParameterKey=GeeSecretArn,ParameterValue="$GEE_SECRET_ARN"
  ParameterKey=NassSecretArn,ParameterValue="$NASS_SECRET_ARN"

)

if [[ -n "${EXISTING_ROLE_ARN}" ]]; then
  PARAMS+=(ParameterKey=ExistingSageMakerExecutionRoleArn,ParameterValue="${EXISTING_ROLE_ARN}")
fi

aws cloudformation deploy   --region "${REGION}"   --stack-name "${STACK}" --template-file infra/cloudformation_full_with_cutoffs.yaml   --capabilities CAPABILITY_NAMED_IAM   --parameter-overrides "${PARAMS[@]}"

echo "Deployed stack ${STACK}"
