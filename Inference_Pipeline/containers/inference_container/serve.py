import os
import io
import json
import numpy as np
import pandas as pd
import mlflow.pyfunc
from flask import Flask, request, Response

MODEL_DIR = "/opt/ml/model"
app = Flask(__name__)

_model = None
_schema = None


def _load_schema():
    path = os.path.join(MODEL_DIR, "feature_schema.json")
    with open(path, "r") as f:
        return json.load(f)


def _prepare(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    expected = schema["expected_features"]
    cat = set(schema.get("categorical_features", []))
    target = schema.get("target")

    X = df.copy()
    if target and target in X.columns:
        X = X.drop(columns=[target])

    # add missing columns
    for c in expected:
        if c not in X.columns:
            X[c] = np.nan

    # enforce order
    X = X[expected]

    # coerce types
    for c in expected:
        if c in cat:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    return X


print("Loading MLflow model from:", MODEL_DIR, flush=True)
_model = mlflow.pyfunc.load_model(MODEL_DIR)
_schema = _load_schema()
print("Model loaded.", flush=True)


@app.get("/ping")
def ping():
    return Response("ok", status=200, mimetype="text/plain")


@app.post("/invocations")
def invocations():
    ct = (request.content_type or "").lower()
    raw = request.data
    if not raw:
        return Response("Empty request body", status=400)

    # Batch Transform sends each S3 object as the request body
    if "parquet" in ct:
        df = pd.read_parquet(io.BytesIO(raw))
    elif "csv" in ct:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    else:
        return Response(f"Unsupported Content-Type: {ct}", status=415)

    if df.empty:
        return Response("Empty input dataframe", status=400)

    # optional: light cleanup
    if "county" in df.columns:
        df["county"] = df["county"].astype(str).str.strip().str.lower()
        df = df[~df["county"].isin(["iowa", "statewide", "all"])]

    # require year for your output
    if "year" not in df.columns:
        return Response("Missing 'year' column", status=400)

    X = _prepare(df, _schema)
    y = np.asarray(_model.predict(X)).reshape(-1)

    out = df[[c for c in ["county", "year"] if c in df.columns]].copy()
    out["prediction"] = y.astype("float64")

    # your current aggregation logic
    out = out.groupby(["county", "year"], as_index=False).agg(prediction=("prediction", "mean"))
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype("int64")

    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    return Response(buf.getvalue(), status=200, mimetype="application/x-parquet")


if __name__ == "__main__":
    # SageMaker expects 8080
    app.run(host="0.0.0.0", port=8080)