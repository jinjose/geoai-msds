import json
from pathlib import Path
import pandas as pd

# geoai/preprocessing/schema.py
import json
from pathlib import Path
from typing import Union, List

def load_feature_list(feature_names_json: Union[str, Path]) -> List[str]:
    p = Path(feature_names_json)  # converts str -> Path safely
    obj = json.loads(p.read_text(encoding="utf-8"))

    # If your JSON is like: {"features": [...]}
    if isinstance(obj, dict) and "features" in obj:
        return obj["features"]

    # If your JSON is like: [...]
    if isinstance(obj, list):
        return obj

    raise ValueError(f"Unexpected JSON format in {p}. Expected list or {{'features': [...]}}")

def enforce_superset_schema(
    df: pd.DataFrame,
    feature_names_json: Path,
    fill_value: float = 0.0,
    keep_first: list[str] | None = None,
) -> pd.DataFrame:
    keep_first = keep_first or ["county", "year", "yield_bu_acre"]

    features = load_feature_list(feature_names_json)

    # Add missing feature columns
    for c in features:
        if c not in df.columns:
            df[c] = fill_value

    # Stable column order: ids/target first (if present), then features, then extras
    ordered = []
    for c in keep_first:
        if c in df.columns and c not in ordered:
            ordered.append(c)
    for c in features:
        if c in df.columns and c not in ordered:
            ordered.append(c)
    for c in df.columns:
        if c not in ordered:
            ordered.append(c)

    return df[ordered]