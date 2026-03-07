#!/usr/bin/env bash
set -euo pipefail

# Build the BYOC inference image and push to ECR.
# Usage:
#   bash scripts/build_and_push_ecr.sh --region ap-south-1 --repo geoai-lgbm-group6-inference --tag v1

REGION=""
REPO=""
TAG="v1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    *) echo "Unknown arg $1"; exit 1 ;;
  esac
done

if [[ -z "${REGION}" || -z "${REPO}" ]]; then
  echo "Usage: --region <region> --repo <ecr-repo-name> [--tag <tag>]"
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"

aws ecr describe-repositories --repository-names "${REPO}" --region "${REGION}" >/dev/null 2>&1 ||   aws ecr create-repository --repository-name "${REPO}" --region "${REGION}" >/dev/null

aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

docker build -f docker/processing/Dockerfile -t "${REPO}:${TAG}" .
docker tag "${REPO}:${TAG}" "${ECR_URI}:${TAG}"
docker push "${ECR_URI}:${TAG}"

echo "Pushed ${ECR_URI}:${TAG}"
