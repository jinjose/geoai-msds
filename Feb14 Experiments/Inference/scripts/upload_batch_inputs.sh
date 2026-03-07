#!/usr/bin/env bash
set -euo pipefail

# Upload feature CSVs into the artifacts bucket under batch/input/
# Usage:
#   bash scripts/upload_batch_inputs.sh --region ap-south-1 --bucket <artifacts-bucket> --input-dir data/features

REGION=""
BUCKET=""
INPUT_DIR="data/features"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg $1"; exit 1 ;;
  esac
done

if [[ -z "${REGION}" || -z "${BUCKET}" ]]; then
  echo "Usage: --region <region> --bucket <bucket> [--input-dir <dir>]"
  exit 1
fi

aws s3 sync "${INPUT_DIR}" "s3://${BUCKET}/batch/input/" --region "${REGION}"
echo "Uploaded inputs to s3://${BUCKET}/batch/input/"
