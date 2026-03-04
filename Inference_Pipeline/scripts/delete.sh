#!/usr/bin/env bash
set -euo pipefail
REGION="${REGION:-ap-south-1}"
STACK="${STACK:-geoai-demo-stack}"
aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
echo "Delete initiated: $STACK"
echo "Note: You may need to empty S3 buckets first."
