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

    for c in expected:
        if c not in X.columns:
            X[c] = np.nan

    X = X[expected]

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


def _read_input_to_df(raw: bytes, content_type: str) -> pd.DataFrame:
    """
    Batch Transform sometimes sends parquet with Content-Type like:
    - application/x-parquet
    - application/octet-stream
    and file extension is not visible to the container.

    We try parquet first when ct hints parquet OR ct is octet-stream,
    fallback to CSV when ct hints csv.
    """
    ct = (content_type or "").lower()

    # Prefer parquet when possible
    if ("parquet" in ct) or ("octet-stream" in ct) or (ct == ""):
        try:
            return pd.read_parquet(io.BytesIO(raw))
        except Exception:
            # fallback to csv if parquet parse fails
            pass

    if "csv" in ct or "text" in ct:
        return pd.read_csv(io.StringIO(raw.decode("utf-8")))

    # last resort: try parquet then csv
    try:
        return pd.read_parquet(io.BytesIO(raw))
    except Exception:
        return pd.read_csv(io.StringIO(raw.decode("utf-8")))


def _ensure_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee df['year'] exists:
    - use 'year' if present
    - else map from 'predict_year' if present
    - else use env var PREDICT_YEAR if set
    """
    if "year" in df.columns:
        return df

    if "predict_year" in df.columns:
        df = df.copy()
        df["year"] = pd.to_numeric(df["predict_year"], errors="coerce")
        return df

    env_py = os.environ.get("PREDICT_YEAR", "").strip()
    if env_py:
        df = df.copy()
        df["year"] = int(env_py)
        return df

    raise ValueError("Missing 'year' (and no 'predict_year' column or PREDICT_YEAR env var to infer it).")


@app.post("/invocations")
def invocations():
    raw = request.data
    if not raw:
        return Response("Empty request body", status=400)

    try:
        df = _read_input_to_df(raw, request.content_type or "")
    except Exception as e:
        return Response(f"Failed to parse input body as parquet/csv: {e}", status=400)

    if df is None or df.empty:
        return Response("Empty input dataframe", status=400)

    # light cleanup
    if "county" in df.columns:
        df = df.copy()
        df["county"] = df["county"].astype(str).str.strip().str.lower()
        df = df[~df["county"].isin(["statewide", "all"])]  # do NOT drop "iowa" if you want statewide in features

    # ensure year exists (explicit output requirement)
    try:
        df = _ensure_year_column(df)
    except Exception as e:
        return Response(str(e), status=400)

    # coerce year robustly
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype("int64")

    X = _prepare(df, _schema)
    y = np.asarray(_model.predict(X)).reshape(-1)

    # Always emit county + year + prediction
    out_cols = []
    if "county" in df.columns:
        out_cols.append("county")
    out_cols.append("year")

    out = df[out_cols].copy()
    out["prediction"] = y.astype("float64")

    # ensure one row per county-year
    if "county" in out.columns:
        out = out.groupby(["county", "year"], as_index=False).agg(prediction=("prediction", "mean"))
    else:
        out = out.groupby(["year"], as_index=False).agg(prediction=("prediction", "mean"))

    buf = io.BytesIO()
    out.to_parquet(buf, index=False, engine="pyarrow")

    # returning parquet bytes is correct; object name is still controlled by SageMaker (.out)
    return Response(buf.getvalue(), status=200, mimetype="application/x-parquet")


if __name__ == "__main__":
    # SageMaker expects 8080
    try:
        app.run(host="0.0.0.0", port=8080)
    except Exception as e:
        print(f"Error starting server: {e}")