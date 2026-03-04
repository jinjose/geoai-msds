#!/usr/bin/env bash
set -euo pipefail

# Orchestrates build + deploy using a JSON config file.
#
# Usage:
#   ./deploy_all.sh --config deploy_config.json
#
# The config controls:
# - which images to build/push (components.*.build_push)
# - whether to create buckets, upload artifacts, and deploy the CFN stack
#   (handled by scripts/deploy.py reading the same config)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${ROOT_DIR}/deploy_config.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: config file not found: $CONFIG_FILE" >&2
  exit 1
fi

BUILD_DIR="${ROOT_DIR}/.build"
IMAGE_URIS_JSON="${BUILD_DIR}/image_uris.json"
mkdir -p "${BUILD_DIR}"

# 1) Build/push selected images (writes .build/image_uris.json)
bash "${ROOT_DIR}/build_push_all.sh" --config "${CONFIG_FILE}"

# 2) Deploy infra/artifacts based on config
python3 "${ROOT_DIR}/deploy.py" --config "${CONFIG_FILE}"

echo ""
# Optional: print state machine ARN if stack exists
REGION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("region","ap-south-1"))' "${CONFIG_FILE}")"
STACK="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("stack","geoai-demo-stack"))' "${CONFIG_FILE}")"

echo "State machine ARN (if deployed):"
aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='StateMachineArn'].OutputValue" --output text 2>/dev/null || true
