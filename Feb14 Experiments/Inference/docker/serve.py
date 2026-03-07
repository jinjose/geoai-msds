#!/usr/bin/env python3
import io
import os
import json
import glob
import logging
import pickle
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, request, Response

# Optional MLflow
try:
    import mlflow.pyfunc
except Exception:
    mlflow = None

# Optional LightGBM
try:
    import lightgbm as lgb
except Exception:
    lgb = None

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", "8080"))
BASE_MODEL_DIR = os.getenv("MODEL_DIR", "/opt/ml/model")

# -------- Runtime cache for multi-model support --------
_MODEL_CACHE: Dict[str, Any] = {}
_FEATURES_CACHE: Dict[str, List[str]] = {}
_CAT_CACHE: Dict[str, List[str]] = {}
_METRICS_CFG_CACHE: Dict[str, Dict[str, Any]] = {}

app = Flask(__name__)


# ----------------------------
# Utilities
# ----------------------------
def _load_json_if_exists(path: str) -> Optional[dict]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _find_first(model_dir: str, patterns: List[str], exclude_substrings: Optional[List[str]] = None) -> Optional[str]:
    exclude_substrings = exclude_substrings or []
    for pat in patterns:
        hits = glob.glob(os.path.join(model_dir, pat))
        hits = [h for h in hits if not any(x in os.path.basename(h) for x in exclude_substrings)]
        if hits:
            hits.sort()
            return hits[0]
    return None


def _get_model_dir(version: str) -> str:
    """
    Multi-model support:
      - default: /opt/ml/model
      - versioned: /opt/ml/model/<version>
    """
    if version and version != "default":
        candidate = os.path.join(BASE_MODEL_DIR, version)
        if os.path.isdir(candidate):
            return candidate
    return BASE_MODEL_DIR


def _load_feature_and_cat_configs(model_dir: str) -> Tuple[List[str], List[str], Dict[str, Any]]:
    # features
    feat_obj = _load_json_if_exists(os.path.join(model_dir, "feature_config.json"))
    if not feat_obj or "features" not in feat_obj:
        raise ValueError(f"Missing feature_config.json with 'features' list in {model_dir}")
    features = feat_obj["features"]
    if not isinstance(features, list) or not features:
        raise ValueError("feature_config.json 'features' must be a non-empty list")

    # categoricals
    cat_obj = _load_json_if_exists(os.path.join(model_dir, "categorical_config.json")) or {}
    cat_cols = cat_obj.get("categorical_columns", [])
    if not isinstance(cat_cols, list):
        raise ValueError("categorical_config.json 'categorical_columns' must be a list")

    # metrics
    metrics_cfg = _load_json_if_exists(os.path.join(model_dir, "metrics_config.json")) or {}
    return features, cat_cols, metrics_cfg


def _load_model(model_dir: str) -> Any:
    """
    Supports:
      - MLflow pyfunc model (directory contains MLmodel)
      - LightGBM Booster model file (model.txt/*.lgb/*.model)
      - Pickle/joblib model file (model.pkl/*.pkl)
    """
    # 1) MLflow pyfunc (preferred if MLmodel present)
    if os.path.exists(os.path.join(model_dir, "MLmodel")):
        if mlflow is None:
            raise RuntimeError("mlflow is required to load MLflow model but is not installed.")
        logger.info("Loading MLflow pyfunc model from %s", model_dir)
        return mlflow.pyfunc.load_model(model_dir)

    # 2) LightGBM Booster file (avoid feature_list.txt etc.)
    booster_path = _find_first(
        model_dir,
        ["model.txt", "model.lgb", "model.model", "*.lgb", "*.model", "*.txt"],
        exclude_substrings=["feature_list", "feature_names", "schema", "config"]
    )
    if booster_path:
        if lgb is None:
            raise RuntimeError("lightgbm is required but not installed.")
        logger.info("Loading LightGBM Booster from %s", booster_path)
        return lgb.Booster(model_file=booster_path)

    # 3) Pickle/joblib
    pkl_path = _find_first(model_dir, ["model.pkl", "*.pkl", "*.pickle", "*.joblib"])
    if pkl_path:
        logger.info("Loading pickled model from %s", pkl_path)
        try:
            import joblib
            return joblib.load(pkl_path)
        except Exception:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)

    raise FileNotFoundError(f"No supported model artifacts found in {model_dir}")


def _get_or_load(version: str) -> Tuple[Any, List[str], List[str], Dict[str, Any], str]:
    """
    Returns (model, features, categorical_cols, metrics_cfg, model_dir)
    cached by version.
    """
    if version in _MODEL_CACHE:
        return (
            _MODEL_CACHE[version],
            _FEATURES_CACHE[version],
            _CAT_CACHE[version],
            _METRICS_CFG_CACHE[version],
            _get_model_dir(version),
        )

    model_dir = _get_model_dir(version)
    features, cat_cols, metrics_cfg = _load_feature_and_cat_configs(model_dir)
    model = _load_model(model_dir)

    _MODEL_CACHE[version] = model
    _FEATURES_CACHE[version] = features
    _CAT_CACHE[version] = cat_cols
    _METRICS_CFG_CACHE[version] = metrics_cfg

    logger.info("Loaded model version=%s dir=%s features=%d categoricals=%s",
                version, model_dir, len(features), cat_cols)
    return model, features, cat_cols, metrics_cfg, model_dir


def _parse_payload_to_df(body: bytes, content_type: str) -> pd.DataFrame:
    """
    Supports:
      - text/csv
      - application/json with dataframe_split
    """
    ctype = (content_type or "text/csv").split(";")[0].strip().lower()

    if ctype in ("text/csv", "application/csv", "application/octet-stream"):
        text = body.decode("utf-8", errors="replace").strip()
        if not text:
            raise ValueError("Empty request body")
        return pd.read_csv(io.StringIO(text))

    if ctype == "application/json":
        obj = json.loads(body.decode("utf-8"))
        if "dataframe_split" in obj:
            dfs = obj["dataframe_split"]
            return pd.DataFrame(dfs["data"], columns=dfs["columns"])
        raise ValueError("Unsupported JSON. Expected {'dataframe_split': {'columns':..., 'data':...}}")

    raise ValueError(f"Unsupported Content-Type: {ctype}. Use text/csv or application/json.")


def _prepare_features(df: pd.DataFrame, features: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    X = df[features].copy()

    # Categoricals (only if you configured them)
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("NA").astype("category")

    # Numeric coercion for non-categoricals
    for c in X.columns:
        if c not in categorical_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # fill NaNs (simple default; can replace with median/imputer if needed)
    X = X.fillna(0)
    return X


def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    # MLflow pyfunc
    if hasattr(model, "predict") and model.__class__.__module__.startswith("mlflow"):
        preds = model.predict(X)
        return np.asarray(preds).reshape(-1)

    # LightGBM Booster
    if lgb is not None and isinstance(model, lgb.Booster):
        preds = model.predict(X)
        return np.asarray(preds).reshape(-1)

    # Pickled estimator
    if hasattr(model, "predict"):
        preds = model.predict(X)
        return np.asarray(preds).reshape(-1)

    raise RuntimeError("Model does not support predict().")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Minimal dependency metrics (no sklearn needed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    # R2
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")

    mae = float(np.mean(np.abs(y_true - y_pred)))

    denom = np.clip(np.abs(y_true), 1e-6, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {"rmse": rmse, "r2": r2, "mae": mae, "mape": mape}


def _format_response(preds: np.ndarray, df_in: pd.DataFrame, metrics: Optional[Dict[str, float]], accept: str) -> Response:
    accept = (accept or "text/csv").split(",")[0].strip().lower()

    if accept in ("application/json", "application/json; charset=utf-8"):
        out = {"predictions": [float(x) for x in preds]}
        if metrics is not None:
            out["metrics"] = metrics
        return Response(json.dumps(out), status=200, mimetype="application/json")

    # Default CSV output (best for Batch Transform)
    # Note: Batch Transform expects plain text/csv response body.
    body = "\n".join(str(float(x)) for x in preds) + "\n"
    # We DO NOT append metrics to CSV body, because Batch Transform treats body as predictions.
    return Response(body, status=200, mimetype="text/csv")


# ----------------------------
# SageMaker endpoints
# ----------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return Response("OK", status=200, mimetype="text/plain")


# Some SageMaker flows call this; returning 200 avoids harmless 404 noise
@app.route("/execution-parameters", methods=["GET"])
def execution_parameters():
    return Response("{}", status=200, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        # Multi-model selection:
        # - header "X-Model-Version: v1" OR env DEFAULT_MODEL_VERSION
        version = request.headers.get("X-Model-Version") or os.getenv("DEFAULT_MODEL_VERSION", "default")
        model, features, cat_cols, metrics_cfg, _ = _get_or_load(version)

        df = _parse_payload_to_df(request.get_data(), request.headers.get("Content-Type", "text/csv"))

        X = _prepare_features(df, features, cat_cols)
        preds = _predict(model, X)

        # Optional metrics if target is present
        metrics = None
        target_col = metrics_cfg.get("target_col", "yield_bu_acre")
        enable_metrics = bool(metrics_cfg.get("enable_metrics_if_target_present", True))

        if enable_metrics and target_col in df.columns:
            y_true = pd.to_numeric(df[target_col], errors="coerce").values
            valid = ~np.isnan(y_true)
            if np.any(valid):
                metrics = _compute_metrics(y_true[valid], preds[valid])

                if metrics_cfg.get("log_metrics_to_cloudwatch", True):
                    logger.info("Metrics (version=%s, n=%d): %s", version, int(np.sum(valid)), metrics)

        # If user wants metrics returned in response (mostly useful for JSON mode / testing)
        if metrics is not None and not metrics_cfg.get("return_metrics_in_response", True):
            metrics = None

        accept = request.headers.get("Accept", "text/csv")
        return _format_response(preds, df, metrics, accept)

    except Exception as e:
        logger.exception("Invocation failed")
        return Response(str(e), status=400, mimetype="text/plain")


def main():
    # Preload default version for faster first request
    default_version = os.getenv("DEFAULT_MODEL_VERSION", "default")
    _get_or_load(default_version)
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()