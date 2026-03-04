# evaluation.py
#
# SageMaker-friendly evaluation script for GeoAI yield inference.
# Scans per-season/per-model prediction Parquet files under:
#   s3://{DATA_BUCKET}/predictions/state_fips=.../county_fips=.../predict_year=.../feature_season=.../run_date=.../model=.../predictions.parquet
#
# Joins with actuals:
#   s3://{DATA_BUCKET}/curated/yield/state_fips=.../county_fips=.../actuals.parquet
#
# Writes outputs to BOTH:
#   1) S3: s3://{DATA_BUCKET}/evaluation/run_date=.../state_fips=.../county_fips=.../predict_year=.../
#   2) Local (for SageMaker Processing): {LOCAL_OUTPUT_DIR}/...

import os
import re
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------
# Environment / Configuration
# ----------------------------
DATA_BUCKET = os.environ["DATA_BUCKET"]
STATE_FIPS = os.environ["STATE_FIPS"]
COUNTY_FIPS = os.environ["COUNTY_FIPS"]
RUN_DATE = os.environ["RUN_DATE"]

# Optional
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
PREDICT_YEAR = os.environ.get("PREDICT_YEAR", "").strip()  # e.g., "2025"
EVAL_YEAR_RANGES = os.environ.get("EVAL_YEAR_RANGES", "").strip()  # JSON list of {name,start,end}
LOCAL_OUTPUT_DIR = os.environ.get("LOCAL_OUTPUT_DIR", "/opt/ml/processing/output").strip()

# Optional filters (comma-separated)
ONLY_SEASONS = {s.strip() for s in os.environ.get("ONLY_SEASONS", "").split(",") if s.strip()}
ONLY_MODELS = {s.strip() for s in os.environ.get("ONLY_MODELS", "").split(",") if s.strip()}

# Safety: exclude accidental statewide rows
BAD_COUNTIES = {"iowa", "statewide", "all", "__statewide__"}

s3 = boto3.client("s3", region_name=AWS_REGION)


# ----------------------------
# Helpers
# ----------------------------
def norm_county(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def list_keys(prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=DATA_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def read_parquet_s3(key: str) -> pd.DataFrame:
    tmp = Path(tempfile.mkdtemp()) / "data.parquet"
    s3.download_file(DATA_BUCKET, key, str(tmp))
    return pd.read_parquet(tmp)


def write_parquet_s3(df: pd.DataFrame, key: str):
    tmp = Path(tempfile.mkdtemp()) / "out.parquet"
    df.to_parquet(tmp, index=False)
    s3.upload_file(str(tmp), DATA_BUCKET, key)


def write_text_s3(text: str, key: str, content_type: str = "application/json"):
    s3.put_object(
        Bucket=DATA_BUCKET,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType=content_type,
    )


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_local_parquet(df: pd.DataFrame, local_path: Path):
    safe_mkdir(local_path.parent)
    df.to_parquet(local_path, index=False)


def write_local_text(text: str, local_path: Path):
    safe_mkdir(local_path.parent)
    local_path.write_text(text, encoding="utf-8")


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_ranges() -> Optional[List[Dict[str, int]]]:
    """
    Expected format:
      EVAL_YEAR_RANGES='[{"name":"hist","start":2013,"end":2020},{"name":"recent","start":2021,"end":2024}]'
    """
    if not EVAL_YEAR_RANGES:
        return None
    try:
        ranges = json.loads(EVAL_YEAR_RANGES)
        if not isinstance(ranges, list):
            return None
        out = []
        for r in ranges:
            if not isinstance(r, dict):
                continue
            if "start" not in r or "end" not in r:
                continue
            out.append(
                {
                    "name": str(r.get("name") or f"{int(r['start'])}_{int(r['end'])}"),
                    "start": int(r["start"]),
                    "end": int(r["end"]),
                }
            )
        return out or None
    except Exception:
        return None


def compute_metrics(dfm: pd.DataFrame, y_true_col: str, y_pred_col: str) -> Tuple[Optional[float], ...]:
    """
    Returns: (MAE, RMSE, MAPE, R2, Bias, N)
    """
    valid = dfm[[y_true_col, y_pred_col]].dropna()
    if valid.empty:
        return (None, None, None, None, None, 0)

    y_true = valid[y_true_col].astype(float).values
    y_pred = valid[y_pred_col].astype(float).values

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    denom = np.maximum(np.abs(y_true), 1e-9)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))

    return (mae, rmse, mape, r2, bias, int(valid.shape[0]))


# ----------------------------
# Main
# ----------------------------
def main():
    # ---- Load actuals ----
    actual_key = f"curated/yield/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/actuals.parquet"
    actual = read_parquet_s3(actual_key)

    if "county" not in actual.columns or "year" not in actual.columns:
        raise ValueError(f"Actuals missing county/year columns. Found: {actual.columns.tolist()}")

    actual["county_norm"] = norm_county(actual["county"])
    actual = actual[~actual["county_norm"].isin(BAD_COUNTIES)].copy()

    y_true_col = find_col(actual, ["yield_bu_acre", "yield", "actual", "y_true"])
    if y_true_col is None:
        raise ValueError(f"Actuals missing yield column. Found: {actual.columns.tolist()}")

    actual_join = actual[["county_norm", "year", y_true_col]].copy()

    # ---- Scan predictions ----
    pred_prefix = f"predictions/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/"

    # Your real layout includes predict_year in the path:
    # .../predict_year=2025/feature_season=aug01/run_date=YYYY-MM-DD/model=.../predictions.parquet
    rx = re.compile(
        r"predict_year=([^/]+)/feature_season=([^/]+)/run_date=([^/]+)/model=([^/]+)/predictions\.parquet$"
    )

    ranges = parse_ranges()

    metrics_rows = []
    comparison_frames = []

    for key in list_keys(pred_prefix):
        if not key.endswith("predictions.parquet"):
            continue

        m = rx.search(key)
        if not m:
            continue

        predict_year, feature_season, run_date, model_name = m.group(1), m.group(2), m.group(3), m.group(4)

        # Filters
        if run_date != RUN_DATE:
            continue
        if PREDICT_YEAR and predict_year != PREDICT_YEAR:
            continue
        if ONLY_SEASONS and feature_season not in ONLY_SEASONS:
            continue
        if ONLY_MODELS and model_name not in ONLY_MODELS:
            continue

        pred = read_parquet_s3(key)

        if "county" not in pred.columns or "year" not in pred.columns:
            raise ValueError(f"Predictions missing county/year columns. Found: {pred.columns.tolist()}")

        pred["county_norm"] = norm_county(pred["county"])
        pred = pred[~pred["county_norm"].isin(BAD_COUNTIES)].copy()

        y_pred_col = find_col(pred, ["y_pred", "prediction", "yhat", "pred"])
        if y_pred_col is None:
            raise ValueError(f"Predictions missing prediction column. Found: {pred.columns.tolist()}")

        merged = pred.merge(actual_join, on=["county_norm", "year"], how="left")

        # Comparison output (per row)
        comp = merged.copy()
        comp["predict_year"] = int(predict_year)
        comp["feature_season"] = feature_season
        comp["model_name"] = model_name
        comp["run_date"] = run_date

        comparison_frames.append(
            comp[["county", "county_norm", "year", y_true_col, y_pred_col, "predict_year", "feature_season", "model_name", "run_date"]]
            .rename(columns={y_true_col: "y_true", y_pred_col: "y_pred"})
        )

        def add_metric_row(range_name: str, dfm: pd.DataFrame):
            mae, rmse, mape, r2, bias, n = compute_metrics(dfm, y_true_col, y_pred_col)
            metrics_rows.append(
                {
                    "range_name": range_name,
                    "predict_year": int(predict_year),
                    "feature_season": feature_season,
                    "model_name": model_name,
                    "run_date": run_date,
                    "N": n,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "R2": r2,
                    "Bias": bias,
                    "predictions_s3_key": key,
                }
            )

        if ranges:
            for r in ranges:
                sub = merged[(merged["year"] >= r["start"]) & (merged["year"] <= r["end"])]
                add_metric_row(r["name"], sub)
        else:
            add_metric_row("all_years", merged)

    metrics = pd.DataFrame(metrics_rows)

    # ---- Output locations ----
    out_prefix = (
        f"evaluation/run_date={RUN_DATE}/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/"
        + (f"predict_year={PREDICT_YEAR}/" if PREDICT_YEAR else "")
    )

    # ---- Write metrics ----
    metrics_key = f"{out_prefix}metrics.parquet"
    write_parquet_s3(metrics, metrics_key)
    print(f"Wrote metrics: s3://{DATA_BUCKET}/{metrics_key}", flush=True)

    # Local outputs for SageMaker Processing (optional but recommended)
    if LOCAL_OUTPUT_DIR:
        local_dir = Path(LOCAL_OUTPUT_DIR)
        write_local_parquet(metrics, local_dir / "metrics.parquet")

    # ---- Ranking JSON (best RMSE per range_name) ----
    ranking = []
    if not metrics.empty and metrics["RMSE"].notna().any():
        best = (
            metrics.dropna(subset=["RMSE"])
            .sort_values(["range_name", "RMSE"], ascending=[True, True])
            .groupby("range_name", as_index=False)
            .first()
        )
        ranking = best.to_dict(orient="records")

        ranking_text = json.dumps(ranking, indent=2)[:200000]
        ranking_key = f"{out_prefix}_RANKING.json"
        write_text_s3(ranking_text, ranking_key)
        print(f"Wrote ranking: s3://{DATA_BUCKET}/{ranking_key}", flush=True)

        if LOCAL_OUTPUT_DIR:
            write_local_text(ranking_text, Path(LOCAL_OUTPUT_DIR) / "_RANKING.json")

    # ---- Write comparison parquet (optional but very useful) ----
    if comparison_frames:
        comparison = pd.concat(comparison_frames, ignore_index=True)
        comp_key = f"{out_prefix}comparison.parquet"
        write_parquet_s3(comparison, comp_key)
        print(f"Wrote comparison: s3://{DATA_BUCKET}/{comp_key}", flush=True)

        if LOCAL_OUTPUT_DIR:
            write_local_parquet(comparison, Path(LOCAL_OUTPUT_DIR) / "comparison.parquet")
    else:
        print("No prediction files matched the scan filters (run_date/predict_year/season/model).", flush=True)

    # Optional: print compact summary
    if ranking:
        print("\nBest models per range (by RMSE):", flush=True)
        for r in ranking:
            print(
                f"- {r['range_name']}: RMSE={r['RMSE']} | season={r['feature_season']} | model={r['model_name']}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR [Evaluation]:", str(e), flush=True)
        sys.exit(1)