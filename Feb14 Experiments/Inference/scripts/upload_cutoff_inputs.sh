#!/usr/bin/env bash
set -euo pipefail

REGION="$1"
BUCKET="$2"
RUN_DATE="$3"
LOCAL_BASE="$4"  # e.g., data/batch_inputs

src="${LOCAL_BASE}/run_date=${RUN_DATE}"

for cutoff in EARLY MID END; do
  f="${src}/cutoff=${cutoff}/features.csv"
  [[ -f "$f" ]] || { echo "Missing $f" >&2; exit 1; }
  aws s3 cp "$f" "s3://${BUCKET}/batch/input/run_date=${RUN_DATE}/cutoff=${cutoff}/features.csv" --region "$REGION"
done

echo "Uploaded cutoff inputs for run_date=${RUN_DATE}"