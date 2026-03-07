# #!/usr/bin/env bash
# set -e
# exec python /opt/program/batch_runner.py
#!/usr/bin/env python3
import io
import os
import json
import glob
import logging
import pickle
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from flask import Flask, Response, request

# LightGBM is optional until we load a Booster model
try:
    import lightgbm as lgb
except Exception:
    lgb = None

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "/opt/ml/model")
PORT = int(os.getenv("PORT", "8080"))

app = Flask(__name__)

_MODEL = None
_MODEL_KIND = None          # "lgb_booster" | "pickle"
_FEATURE_NAMES: Optional[List[str]] = None


def _find_first(patterns: List[str], exclude_substrings: Optional[List[str]] = None) -> Optional[str]:
    exclude_substrings = exclude_substrings or []
    for pat in patterns:
        hits = glob.glob(os.path.join(MODEL_DIR, pat))
        hits = [h for h in hits if not any(x in os.path.basename(h) for x in exclude_substrings)]
        if hits:
            hits.sort()
            return hits[0]
    return None



def _load_model_and_metadata() -> Tuple[object, str, Optional[List[str]]]:
    """
    Loads a model from /opt/ml/model.
    Supported:
      - LightGBM Booster text model (*.txt, *.lgb, *.model)
      - Pickled model (*.pkl, *.pickle) (LightGBM sklearn wrapper or any estimator)
      - Joblib (*.joblib)
    Optional feature names:
      - feature_names.json in MODEL_DIR: {"features": [...]}
    """
    # Optional explicit feature list
    feature_json = os.path.join(MODEL_DIR, "feature_names.json")
    feature_names = None
    if os.path.exists(feature_json):
        try:
            with open(feature_json, "r", encoding="utf-8") as f:
                feature_names = json.load(f).get("features")
            if feature_names:
                logger.info("Loaded feature names from feature_names.json (%d features)", len(feature_names))
        except Exception as e:
            logger.warning("Could not read feature_names.json: %s", e)

    # 1) LightGBM Booster text model
    # 1) LightGBM Booster text model
    booster_path = _find_first(
        ["model.txt", "model.lgb", "model.model", "*.lgb", "*.model", "*.txt"],
        exclude_substrings=["feature_list", "feature_names"]
    )

    if booster_path:
        if lgb is None:
            raise RuntimeError("LightGBM is required but not importable in this image.")
        booster = lgb.Booster(model_file=booster_path)
        if not feature_names:
            try:
                feature_names = booster.feature_name()
            except Exception:
                feature_names = None
        logger.info("Loaded LightGBM Booster from %s", booster_path)
        return booster, "lgb_booster", feature_names

    # 2) pickle / joblib
    pkl_path = _find_first(["*.pkl", "*.pickle", "*.joblib"])
    if pkl_path:
        # joblib is typically installed; but if not, fall back to pickle.
        try:
            import joblib
            model = joblib.load(pkl_path)
        except Exception:
            with open(pkl_path, "rb") as f:
                model = pickle.load(f)
        # Try to infer feature names from sklearn-style estimators
        if not feature_names:
            for attr in ("feature_names_in_",):
                if hasattr(model, attr):
                    feature_names = list(getattr(model, attr))
                    break
        logger.info("Loaded pickled model from %s", pkl_path)
        return model, "pickle", feature_names

    raise FileNotFoundError(
        f"No supported model file found in {MODEL_DIR}. "
        "Expected one of: *.txt/*.lgb/*.model or *.pkl/*.pickle/*.joblib"
    )


def _parse_csv_payload(payload: bytes) -> pd.DataFrame:
    """
    Accepts CSV payload that may be:
      - single row without header (SingleRecord)
      - multiple rows with or without header (MultiRecord)
    Returns a DataFrame.
    """
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("Empty request body")

    # If SageMaker sends a single line, it may not include a newline.
    # We'll try reading with header inference first.
    buf = io.StringIO(text)

    # Attempt: detect header (pandas will treat first row as header by default only if header=0)
    # But we don't know if header exists; we try both.
    try:
        df_try_header = pd.read_csv(buf)
        # Reset buffer for second attempt
        buf.seek(0)
        df_try_no_header = pd.read_csv(buf, header=None)
    except Exception as e:
        raise ValueError(f"Unable to parse CSV: {e}") from e

    # Heuristic:
    # If header-read produces column names that look like real feature names (not ints),
    # and has same number of columns as no-header parse, prefer header version.
    if df_try_header.shape[1] == df_try_no_header.shape[1]:
        # if header columns are not purely numeric like 0,1,2...
        header_cols = list(df_try_header.columns)
        looks_like_header = any(isinstance(c, str) for c in header_cols)
        if looks_like_header:
            return df_try_header

    return df_try_no_header


def _align_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align df columns to model feature names if we have them.
    - If df has no header and feature names exist: assign them.
    - If df has header: reorder/validate.
    """
    global _FEATURE_NAMES

    if not _FEATURE_NAMES:
        # No feature metadata; just return numeric values as-is
        return df

    # If df columns are integers (0..n-1), assume no header
    if all(isinstance(c, int) for c in df.columns):
        if df.shape[1] != len(_FEATURE_NAMES):
            raise ValueError(
                f"Input has {df.shape[1]} columns but model expects {len(_FEATURE_NAMES)} features."
            )
        df.columns = _FEATURE_NAMES
        return df

    # Has header: ensure all required columns exist
    missing = [c for c in _FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Reorder and drop extras
    return df[_FEATURE_NAMES]


def _predict(df: pd.DataFrame) -> np.ndarray:
    """
    Produces prediction array shape (n,).
    """
    global _MODEL, _MODEL_KIND
    X = df

    if _MODEL_KIND == "lgb_booster":
        # LightGBM Booster expects numeric matrix; pandas ok
        preds = _MODEL.predict(X)
        return np.asarray(preds).reshape(-1)

    # Pickled estimator
    if hasattr(_MODEL, "predict"):
        preds = _MODEL.predict(X)
        return np.asarray(preds).reshape(-1)

    raise RuntimeError("Loaded model does not have a supported predict interface.")


@app.route("/ping", methods=["GET"])
def ping():
    return Response(response="OK", status=200, mimetype="text/plain")


@app.route("/invocations", methods=["POST"])
def invocations():
    global _MODEL, _MODEL_KIND, _FEATURE_NAMES

    ctype = request.headers.get("Content-Type", "text/csv").split(";")[0].strip().lower()
    if ctype not in ("text/csv", "application/csv", "application/octet-stream"):
        return Response(
            response=f"Unsupported Content-Type: {ctype}. Use text/csv.",
            status=415,
            mimetype="text/plain",
        )

    try:
        payload = request.get_data()
        df = _parse_csv_payload(payload)
        df = _align_features(df)

        # Convert everything to numeric where possible
        # (If you have categorical encoding already, keep it numeric in input.)
        df = df.apply(pd.to_numeric, errors="ignore")

        preds = _predict(df)

        # Return as CSV: one prediction per line
        out = "\n".join(str(float(p)) for p in preds) + "\n"
        return Response(response=out, status=200, mimetype="text/csv")

    except Exception as e:
        logger.exception("Invocation failed")
        return Response(response=str(e), status=400, mimetype="text/plain")


def main():
    global _MODEL, _MODEL_KIND, _FEATURE_NAMES
    _MODEL, _MODEL_KIND, _FEATURE_NAMES = _load_model_and_metadata()
    logger.info("Model loaded. kind=%s, features=%s",
                _MODEL_KIND, len(_FEATURE_NAMES) if _FEATURE_NAMES else "unknown")

    # IMPORTANT: SageMaker expects the container to listen on 0.0.0.0:8080
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
