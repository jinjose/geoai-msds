import os
import json
import sys
import pandas as pd
import numpy as np
import mlflow.pyfunc
import boto3
import io, zipfile

MODEL_DIR = "/opt/ml/model"
MODEL_META_S3_KEY = os.environ.get("MODEL_META_S3_KEY","").strip()
MODEL_META_S3_URI = os.environ.get("MODEL_META_S3_URI","").strip()
AWS_REGION = os.environ.get("AWS_REGION","ap-south-1")

INPUT_DIR = "/opt/ml/input/data"
OUTPUT_DIR = "/opt/ml/output"


def parse_s3_uri(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {uri}")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key

def load_feature_schema() -> dict | None:
    if not MODEL_META_S3_KEY and not MODEL_META_S3_URI:
        return None
    s3 = boto3.client("s3", region_name=AWS_REGION)
    if MODEL_META_S3_URI:
        b, k = parse_s3_uri(MODEL_META_S3_URI)
    else:
        # In transform, prefer full key passed. Bucket may be MODEL_META_S3_BUCKET else use same as DATA_BUCKET env.
        b = os.environ.get("DATA_BUCKET") or parse_s3_uri(os.environ.get("MODEL_ARTIFACTS_S3","s3://")).__getitem__(0) if os.environ.get("MODEL_ARTIFACTS_S3") else None
        if not b:
            raise RuntimeError("Need DATA_BUCKET (or MODEL_META_S3_URI) to download schema")
        k = MODEL_META_S3_KEY
    obj = s3.get_object(Bucket=b, Key=k)
    body = obj["Body"].read()
    if k.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(body), "r") as z:
            with z.open("feature_schema.json") as f:
                return json.load(f)
    return json.loads(body.decode("utf-8"))

def get_expected_features(schema: dict) -> list[str]:
    feats = schema.get("expected_features", [])
    target = schema.get("target")
    feats = [f for f in feats if f != target]
    return feats

def find_input_file():
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            lf = f.lower()
            if lf.endswith(".parquet") or lf.endswith(".pq") or lf.endswith(".csv"):
                return os.path.join(root, f)
    raise RuntimeError("No input file found under /opt/ml/input/data")

def read_input(path: str) -> pd.DataFrame:
    lp = path.lower()
    if lp.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)

def write_output(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)

def load_model_if_needed(model_type: str):
    if model_type == "mlflow":
        return mlflow.pyfunc.load_model(MODEL_DIR)
    return None

def main():
    model_name = os.environ.get("MODEL_NAME", "").strip()
    model_type = os.environ.get("MODEL_TYPE", "").strip().lower()

    # If MODEL_TYPE not provided, infer baseline by name
    if not model_type:
        model_type = "native" if model_name.lower() in {"lag1_baseline","lag-1-baseline","lag1-baseline"} else "mlflow"

    in_path = find_input_file()
    df = read_input(in_path)

    # Contract: county,year + features (NO yield_bu_acre)
    for col in ["yield_bu_acre","y_true"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    #id_cols = [c for c in ["county","year"] if c in df.columns]
    id_cols = [c for c in ["geoid","county","county_name","year"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in id_cols]

    if model_type == "native":
        # Lag-1 baseline uses lag1_yield feature (Option A)
        if "lag1_yield" not in df.columns:
            raise ValueError("lag1_yield not found for baseline prediction")
        y_pred = df["lag1_yield"].astype(float).values
    else:
        model = load_model_if_needed("mlflow")
        X = df[feature_cols]
        y_pred = model.predict(X)

    out = df[id_cols].copy() if id_cols else pd.DataFrame()
    out["y_pred"] = np.array(y_pred, dtype=float)
    out["model_name"] = model_name or ("lag1_baseline" if model_type=="native" else "mlflow_model")

    out_path = os.path.join(OUTPUT_DIR, "predictions.parquet")
    write_output(out, out_path)
    print("Wrote:", out_path, flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, flush=True)
        sys.exit(1)
