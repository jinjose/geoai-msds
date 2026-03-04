#!/usr/bin/env bash
set -euo pipefail

# Build/push containers selectively based on a JSON config file.
#
# Usage:
#   ./build_push_all.sh --config deploy_config.json
#
# Notes:
# - Config booleans control which images are built/pushed:
#     components.ingestion.build_push, components.feature.build_push, ...
# - If a component build_push=false, its image URI must come from either:
#     images.<name>_image_uri in config, OR an existing .build/image_uris.json
#
# Config format example: deploy_config.example.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${ROOT_DIR}/deploy_config.json"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

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

pyget() {
  python3 - "$CONFIG_FILE" "$1" <<'PY'
import json, sys
path = sys.argv[1]
expr = sys.argv[2]
cfg = json.load(open(path))
print(eval(expr, {"cfg": cfg}))
PY
}

REGION="$(pyget 'cfg.get("region","ap-south-1")')"
PROJECT="$(pyget 'cfg.get("project","geoai-demo")')"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BASE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

BUILD_DIR="${ROOT_DIR}/.build"
IMAGE_URIS_JSON="${BUILD_DIR}/image_uris.json"
mkdir -p "${BUILD_DIR}"

# Read build flags from config (default true)
DEPLOY_INGESTION="$(pyget 'cfg.get("components",{}).get("ingestion",{}).get("build_push", True)')"
DEPLOY_FEATURE="$(pyget 'cfg.get("components",{}).get("feature",{}).get("build_push", True)')"
DEPLOY_INFERENCE="$(pyget 'cfg.get("components",{}).get("inference",{}).get("build_push", True)')"
DEPLOY_EVALUATION="$(pyget 'cfg.get("components",{}).get("evaluation",{}).get("build_push", True)')"

# Image URI overrides from config (optional)
CFG_INGESTION_URI="$(pyget 'cfg.get("images",{}).get("ingestion_image_uri","")')"
CFG_FEATURE_URI="$(pyget 'cfg.get("images",{}).get("feature_image_uri","")')"
CFG_INFERENCE_URI="$(pyget 'cfg.get("images",{}).get("inference_image_uri","")')"
CFG_EVALUATION_URI="$(pyget 'cfg.get("images",{}).get("evaluation_image_uri","")')"

# Load existing URIs (so skipping a component doesn't break deploy)
existing_ingestion=""
existing_feature=""
existing_inference=""
existing_evaluation=""
if [[ -f "${IMAGE_URIS_JSON}" ]]; then
  existing_ingestion="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("IngestionImageUri",""))' "${IMAGE_URIS_JSON}")"
  existing_feature="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("FeatureImageUri",""))' "${IMAGE_URIS_JSON}")"
  existing_inference="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("InferenceImageUri",""))' "${IMAGE_URIS_JSON}")"
  existing_evaluation="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("EvaluationImageUri",""))' "${IMAGE_URIS_JSON}")"
fi

is_true() {
  # Accept python bool prints: True/False
  case "${1,,}" in
    true|1|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_repo() {
  local repo="$1"
  aws ecr describe-repositories --repository-names "$repo" --region "$REGION" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "$repo" --region "$REGION" >/dev/null
}

did_login=false
maybe_login() {
  if [[ "$did_login" == "false" ]]; then
    aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$BASE"
    did_login=true
  fi
}

build_push() {
  local name="$1"   # ingestion|feature|inference|evaluation
  local ctx="$2"    # folder under containers/
  local repo="${PROJECT}-${name}"

  ensure_repo "${repo}"

  local local_tag="${PROJECT}-${name}:latest"
  local remote_tag="${BASE}/${PROJECT}-${name}:latest"
  local context_path="${REPO_ROOT}/containers/${ctx}"


  if [[ ! -d "$context_path" ]]; then
    echo "ERROR: Build context folder not found: $context_path" >&2
    exit 1
  fi

  maybe_login
  echo "==> Building ${local_tag} from ${context_path}"
  docker build -t "${local_tag}" "${context_path}"
  docker tag "${local_tag}" "${remote_tag}"
  echo "==> Pushing ${remote_tag}"
  docker push "${remote_tag}"

  echo "${remote_tag}"
}

# Final URIs (prefer config overrides, else existing)
INGESTION_IMAGE_URI="${CFG_INGESTION_URI:-$existing_ingestion}"
FEATURE_IMAGE_URI="${CFG_FEATURE_URI:-$existing_feature}"
INFERENCE_IMAGE_URI="${CFG_INFERENCE_URI:-$existing_inference}"
EVALUATION_IMAGE_URI="${CFG_EVALUATION_URI:-$existing_evaluation}"

if is_true "$DEPLOY_INGESTION"; then
  INGESTION_IMAGE_URI="$(build_push ingestion ingestion_container)"
fi
if is_true "$DEPLOY_FEATURE"; then
  FEATURE_IMAGE_URI="$(build_push feature feature_container)"
fi
if is_true "$DEPLOY_INFERENCE"; then
  INFERENCE_IMAGE_URI="$(build_push inference inference_container)"
fi
if is_true "$DEPLOY_EVALUATION"; then
  EVALUATION_IMAGE_URI="$(build_push evaluation evaluation_container)"
fi

missing=0
[[ -n "$INGESTION_IMAGE_URI" ]] || { echo "ERROR: Missing ingestion image URI. Provide images.ingestion_image_uri in config OR keep .build/image_uris.json OR set components.ingestion.build_push=true" >&2; missing=1; }
[[ -n "$FEATURE_IMAGE_URI" ]] || { echo "ERROR: Missing feature image URI. Provide images.feature_image_uri in config OR keep .build/image_uris.json OR set components.feature.build_push=true" >&2; missing=1; }
[[ -n "$INFERENCE_IMAGE_URI" ]] || { echo "ERROR: Missing inference image URI. Provide images.inference_image_uri in config OR keep .build/image_uris.json OR set components.inference.build_push=true" >&2; missing=1; }
[[ -n "$EVALUATION_IMAGE_URI" ]] || { echo "ERROR: Missing evaluation image URI. Provide images.evaluation_image_uri in config OR keep .build/image_uris.json OR set components.evaluation.build_push=true" >&2; missing=1; }
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

cat > "${IMAGE_URIS_JSON}" <<EOF
{
  "IngestionImageUri": "${INGESTION_IMAGE_URI}",
  "FeatureImageUri": "${FEATURE_IMAGE_URI}",
  "InferenceImageUri": "${INFERENCE_IMAGE_URI}",
  "EvaluationImageUri": "${EVALUATION_IMAGE_URI}"
}
EOF

echo "Wrote ${IMAGE_URIS_JSON}"
cat "${IMAGE_URIS_JSON}"
