#!/usr/bin/env bash
set -euo pipefail

# Package lambda/transform_scheduler.py into a zip and upload to S3 for CloudFormation.
# Usage:
#   bash scripts/package_lambda.sh --region ap-south-1 --bucket <cfn-assets-bucket> --key lambda/transform_scheduler.zip

REGION=""
BUCKET=""
KEY="lambda/transform_scheduler.zip"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    *) echo "Unknown arg $1"; exit 1 ;;
  esac
done

if [[ -z "${REGION}" || -z "${BUCKET}" ]]; then
  echo "Usage: --region <region> --bucket <bucket> [--key <key>]"
  exit 1
fi

TMPDIR="$(mktemp -d)"
cp -r lambda "$TMPDIR/lambda"
pushd "$TMPDIR" >/dev/null
zip -r transform_scheduler.zip lambda/transform_scheduler.py >/dev/null
aws s3 cp transform_scheduler.zip "s3://${BUCKET}/${KEY}" --region "${REGION}"
popd >/dev/null
rm -rf "$TMPDIR"

echo "Uploaded lambda zip to s3://${BUCKET}/${KEY}"
