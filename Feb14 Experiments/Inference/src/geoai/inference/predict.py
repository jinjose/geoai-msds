
"""LightGBM inference utilities for SageMaker Batch Transform container.

Design:
- SageMaker Batch Transform mounts:
    /opt/ml/input/data/<channel>/...  (input)
    /opt/ml/model/                    (model artifacts from model.tar.gz)
    /opt/ml/output/                   (output)
- We treat input as CSV with header.
- Output is CSV with appended columns: y_pred and prediction_timestamp_utc
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import lightgbm as lgb
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "lightgbm is required inside the inference image. Ensure it is in docker/requirements.txt"
    ) from e


@dataclass(frozen=True)
class PredictionResult:
    df_out: pd.DataFrame
    model_path: Path
    feature_columns: List[str]


def _default_model_dir() -> Path:
    return Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


def load_booster(model_dir: Optional[Path] = None) -> tuple[lgb.Booster, Path]:
    model_dir = model_dir or _default_model_dir()
    # Support common names
    candidates = [
        model_dir / "model.pkl",
        model_dir / "model.txt",
        model_dir / "model.lgb",
        model_dir / "model.bin",
        model_dir / "model.json",
    ]
    for p in candidates:
        if p.exists():
            # Booster can load from file; for pickle, try joblib
            if p.suffix == ".pkl":
                import joblib
                booster = joblib.load(p)
                if not isinstance(booster, lgb.Booster):
                    raise TypeError(f"model.pkl is not a lightgbm.Booster (got {type(booster)})")
                return booster, p
            booster = lgb.Booster(model_file=str(p))
            return booster, p

    raise FileNotFoundError(
        f"No model file found in {model_dir}. Expected one of: "
        + ", ".join(str(c.name) for c in candidates)
    )


def predict_csv(
    input_csv: Path,
    output_csv: Path,
    id_columns: Optional[List[str]] = None,
    model_dir: Optional[Path] = None,
    expected_feature_list_path: Optional[Path] = None,
) -> PredictionResult:
    booster, model_path = load_booster(model_dir=model_dir)

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input file {input_csv} has no rows.")

    # Optional: enforce feature contract
    feature_cols = [c for c in df.columns]
    if id_columns:
        feature_cols = [c for c in feature_cols if c not in set(id_columns)]

    if expected_feature_list_path and expected_feature_list_path.exists():
        expected = [x.strip() for x in expected_feature_list_path.read_text().splitlines() if x.strip()]
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected and (not id_columns or c not in id_columns)]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")
        # Allow extras but log by adding a column
        feature_cols = [c for c in expected if (not id_columns or c not in id_columns)]

    X = df[feature_cols]
    y_pred = booster.predict(X)

    df_out = df.copy()
    df_out["y_pred"] = y_pred
    df_out["prediction_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    return PredictionResult(df_out=df_out, model_path=model_path, feature_columns=feature_cols)
