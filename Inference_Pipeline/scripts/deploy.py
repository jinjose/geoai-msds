import argparse, json, subprocess, zipfile
from pathlib import Path
import os

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "infra" / "template.yaml"
LAMBDA_SRC = ROOT / "src" / "lambdas"

def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

def is_true(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y")

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_bucket(bucket, region):
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except ClientError:
        pass
    if region == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})

def zip_lambda(src_dir: Path, out_zip: Path):
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_dir():
                continue
            z.write(p, p.relative_to(src_dir).as_posix())

def upload_file(bucket, key, path, region):
    s3 = boto3.client("s3", region_name=region)
    s3.upload_file(str(path), bucket, key)

def put_json(bucket, key, data, region):
    s3 = boto3.client("s3", region_name=region)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def template_has(token: str) -> bool:
    try:
        txt = TEMPLATE.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return token in txt

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", required=True, help="Path to deploy_config.json")

    # Optional CLI overrides (take precedence over config)
    ap.add_argument("--project")
    ap.add_argument("--region")
    ap.add_argument("--stack")
    ap.add_argument("--data-bucket")
    ap.add_argument("--artifacts-bucket")

    # Optional overrides for images
    ap.add_argument("--ingestion-image-uri")
    ap.add_argument("--feature-image-uri")
    ap.add_argument("--inference-image-uri")
    ap.add_argument("--evaluation-image-uri")

    args = ap.parse_args()
    cfg = load_cfg(args.config)

    project = args.project or cfg.get("project", "geoai-demo")
    region = args.region or cfg.get("region", "ap-south-1")
    stack = args.stack or cfg.get("stack", "geoai-demo-stack")

    buckets = cfg.get("buckets", {})
    artifacts_cfg = cfg.get("artifacts", {})
    pipeline = cfg.get("pipeline", {})
    deploy_cfg = cfg.get("deploy", {})
    images_cfg = cfg.get("images", {})

    data_bucket = args.data_bucket or buckets.get("data_bucket")
    artifacts_bucket = args.artifacts_bucket or buckets.get("artifacts_bucket")

    if not data_bucket or not artifacts_bucket:
        raise SystemExit("ERROR: config must provide buckets.data_bucket and buckets.artifacts_bucket (or pass via CLI).")

    artifacts_prefix = artifacts_cfg.get("prefix", "lambda-artifacts")

    create_data_bucket = buckets.get("create_data_bucket", True)
    create_artifacts_bucket = buckets.get("create_artifacts_bucket", True)
    upload_lambda_artifacts = artifacts_cfg.get("upload_lambda_artifacts", True)
    upload_model_registry = artifacts_cfg.get("upload_model_registry", True)
    deploy_stack = deploy_cfg.get("deploy_stack", True)

    schedule = pipeline.get("schedule", "rate(1 day)")
    enable_schedule = pipeline.get("enable_schedule", False)
    backfill_start_year = pipeline.get("backfill_start_year", 2014)

    # Images: from CLI override > config
    ingestion_image_uri = args.ingestion_image_uri or images_cfg.get("ingestion_image_uri")
    feature_image_uri = args.feature_image_uri or images_cfg.get("feature_image_uri")
    inference_image_uri = args.inference_image_uri or images_cfg.get("inference_image_uri")
    evaluation_image_uri = args.evaluation_image_uri or images_cfg.get("evaluation_image_uri")

    # Also allow reading from .build/image_uris.json if config didn't include them
    image_uris_path = ROOT / ".build" / "image_uris.json"
    if image_uris_path.exists():
        d = json.load(open(image_uris_path, "r", encoding="utf-8"))
        ingestion_image_uri = ingestion_image_uri or d.get("IngestionImageUri")
        feature_image_uri = feature_image_uri or d.get("FeatureImageUri")
        inference_image_uri = inference_image_uri or d.get("InferenceImageUri")
        evaluation_image_uri = evaluation_image_uri or d.get("EvaluationImageUri")

    # 1) Buckets
    if create_data_bucket:
        ensure_bucket(data_bucket, region)
    else:
        print(f"==> Skipping data bucket creation: {data_bucket}")

    if create_artifacts_bucket:
        ensure_bucket(artifacts_bucket, region)
    else:
        print(f"==> Skipping artifacts bucket creation: {artifacts_bucket}")

    build_dir = ROOT / ".build"
    build_dir.mkdir(exist_ok=True)

    # 2) Package + upload lambda artifacts
    if upload_lambda_artifacts:
        packages = [
            ("model_registry_loader", "model_registry_loader.zip"),
            ("eval_lambda", "eval_lambda.zip"),
            ("prediction_rename_lambda", "prediction_rename_lambda.zip"),
        ]
        for folder, zipname in packages:
            src = LAMBDA_SRC / folder
            out = build_dir / zipname
            zip_lambda(src, out)
            upload_file(artifacts_bucket, f"{artifacts_prefix}/{zipname}", out, region)
        print(f"==> Uploaded lambda artifacts to s3://{artifacts_bucket}/{artifacts_prefix}/")
    else:
        print("==> Skipping lambda packaging/upload")

    # 3) Upload initial model registry
    if upload_model_registry:
        registry = {
            "models": [
                {"name": "lag1_baseline", "type": "native"},
                {"name": "LightGBM-lag", "type": "mlflow", "model_s3_uri": f"s3://{data_bucket}/models/LightGBM-lag/"},
                {"name": "Ridge", "type": "mlflow", "model_s3_uri": f"s3://{data_bucket}/models/Ridge/"},
                {"name": "LightGBM-No-Lag", "type": "mlflow", "model_s3_uri": f"s3://{data_bucket}/models/LightGBM-No-Lag/"},
            ]
        }
        put_json(data_bucket, "config/model_registry.json", registry, region)
        print(f"==> Wrote model registry: s3://{data_bucket}/config/model_registry.json")
    else:
        print("==> Skipping model registry upload")

    # 4) CloudFormation deploy
    if not deploy_stack:
        print("==> Skipping CloudFormation deploy (deploy.deploy_stack=false)")
        return

    # Template still needs all 4 image URIs
    missing = [k for k,v in {
        "images.ingestion_image_uri": ingestion_image_uri,
        "images.feature_image_uri": feature_image_uri,
        "images.inference_image_uri": inference_image_uri,
        "images.evaluation_image_uri": evaluation_image_uri,
    }.items() if not v]
    if missing:
        raise SystemExit("ERROR: Missing required image URIs: " + ", ".join(missing) +
                         ". Provide them in config.images.* or run build_push to generate .build/image_uris.json.")

    params = [
        f"ProjectName={project}",
        f"DataBucketName={data_bucket}",
        f"ArtifactsBucketName={artifacts_bucket}",
        f"LambdaArtifactsPrefix={artifacts_prefix}",
        f"PipelineScheduleExpression={schedule}",
        f"PipelineEnabled={'true' if enable_schedule else 'false'}",
        f"BackfillStartYear={backfill_start_year}",
        f"IngestionImageUri={ingestion_image_uri}",
        f"FeatureImageUri={feature_image_uri}",
        f"InferenceImageUri={inference_image_uri}",
        f"EvaluationImageUri={evaluation_image_uri}",
    ]

    # Optional CFN conditional params (only if template supports them)
    comps = cfg.get("components", {})
    optional = [
        ("DeployIngestion", "ingestion"),
        ("DeployFeature", "feature"),
        ("DeployInference", "inference"),
        ("DeployEvaluation", "evaluation"),
    ]
    for key, comp in optional:
        if template_has(key):
            # default true unless explicitly false
            val = comps.get(comp, {}).get("deploy_resource", True)
            params.append(f"{key}={'true' if val else 'false'}")

    run(
        [
            "aws",
            "cloudformation",
            "deploy",
            "--region",
            region,
            "--stack-name",
            stack,
            "--template-file",
            str(TEMPLATE),
            "--capabilities",
            "CAPABILITY_NAMED_IAM",
            "--parameter-overrides",
            *params,
        ]
    )

    print(f"\nDeployed. Model registry at: s3://{data_bucket}/config/model_registry.json")

if __name__ == "__main__":
    main()
