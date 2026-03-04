import json
import os
import boto3

s3 = boto3.client("s3")

DATA_BUCKET = os.environ["DATA_BUCKET"]
REGISTRY_KEY = os.environ.get("REGISTRY_KEY", "config/model_registry.json")

# Where the SageMaker ModelDataUrl tarball lives under each model_s3_prefix
# If your tarball name differs, set env MODEL_TARBALL_KEY accordingly.
MODEL_TARBALL_KEY = os.environ.get("MODEL_TARBALL_KEY", "model.tar.gz")

def _normalize_season_from_model_name(name: str) -> str:
    # Example: "Jun01_LightGBM-..." -> "jun01"
    head = name.split("_", 1)[0]
    return head.strip().lower()

def lambda_handler(event, context):
    obj = s3.get_object(Bucket=DATA_BUCKET, Key=REGISTRY_KEY)
    registry = json.loads(obj["Body"].read().decode("utf-8"))

    models = registry.get("models", [])
    if not models:
        raise ValueError("model_registry.json has no models")

    # Validate + derive model_s3_uri when only model_s3_prefix exists
    for m in models:
        if "name" not in m or "type" not in m:
            raise ValueError(f"Invalid model entry (needs name,type): {m}")

        if m["type"] == "mlflow":
            # Allow either model_s3_uri OR model_s3_prefix
            if "model_s3_uri" not in m:
                if "model_s3_prefix" not in m:
                    raise ValueError(f"MLflow model missing model_s3_prefix/model_s3_uri: {m}")
                prefix = m["model_s3_prefix"].rstrip("/")
                m["model_s3_uri"] = f"{prefix}/{MODEL_TARBALL_KEY}"

        # Attach season key for grouping
        m["_season"] = _normalize_season_from_model_name(m["name"])

    # OPTIONAL FILTER
    model_filter = event.get("model_filter")
    if model_filter:
        if not isinstance(model_filter, list) or not all(isinstance(x, str) for x in model_filter):
            raise ValueError("model_filter must be a list of strings")
        wanted = {x.strip() for x in model_filter if x.strip()}
        models = [m for m in models if m["name"] in wanted]
        if not models:
            raise ValueError(f"model_filter removed all models. Requested: {model_filter}")

    # Build seasons grid the state machine expects for inference
    seasons_in = event.get("seasons", [])
    if not seasons_in:
        raise ValueError("Input is missing seasons")

    # Group models by season
    models_by_season = {}
    for m in models:
        models_by_season.setdefault(m["_season"], []).append(
            {"name": m["name"], "model_s3_uri": m["model_s3_uri"]}
        )

    seasons_out = []
    for s in seasons_in:
        # s is expected to be an object (per your updated input)
        if not isinstance(s, dict) or "feature_season" not in s:
            raise ValueError(f"Each seasons[] item must be an object with feature_season. Got: {s}")

        feature_season = s["feature_season"].strip().lower()
        if feature_season not in models_by_season:
            raise ValueError(f"No models registered for season={feature_season}")
        seasons_out.append({
            "feature_season": feature_season,
            "model_name": s.get("model_name", ""),
            "model_s3_prefix": s.get("model_s3_prefix", ""),
            "models": models_by_season.get(feature_season, [])
        })

    # IMPORTANT: return the structure that RunSeasonModelGrid expects
    return {"seasons": seasons_out}