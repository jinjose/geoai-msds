import os
import re
import json
import sys
import boto3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import tempfile

DATA_BUCKET = os.environ["DATA_BUCKET"]
STATE_FIPS = os.environ["STATE_FIPS"]
COUNTY_FIPS = os.environ["COUNTY_FIPS"]
RUN_DATE = os.environ["RUN_DATE"]
EVAL_YEAR_RANGES = os.environ.get("EVAL_YEAR_RANGES", "").strip()  # JSON list

AWS_REGION = os.environ.get("AWS_REGION","ap-south-1")

s3 = boto3.client("s3", region_name=AWS_REGION)

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

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return mae, rmse, mape, r2, bias

def main():
    actual_key = f"curated/yield/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/actuals.parquet"
    actual = read_parquet_s3(actual_key)

    pred_prefix = f"predictions/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/"
    rows = []
    comp_rows = []

    ranges = None
    if EVAL_YEAR_RANGES:
        try:
            ranges = json.loads(EVAL_YEAR_RANGES)
        except Exception:
            ranges = None


    for key in list_keys(pred_prefix):
        if not key.endswith("predictions.parquet"):
            continue
        m = re.search(r"feature_season=([^/]+)/run_date=([^/]+)/model=([^/]+)/predictions\.parquet$", key)
        if not m:
            continue
        feature_season, run_date, model_name = m.group(1), m.group(2), m.group(3)
        if run_date != RUN_DATE:
            continue

        pred = read_parquet_s3(key)
        merged = pred.merge(actual, on=["county","year"], how="left")

        merged_out = merged.copy()
        merged_out["feature_season"] = feature_season
        merged_out["model_name"] = model_name
        merged_out["run_date"] = run_date
        comp_rows.append(merged_out[["county","year","yield_bu_acre","y_pred","feature_season","model_name","run_date"]])

        def _metric_row(rname, dfm):
            if dfm["yield_bu_acre"].notna().any():
                y_true = dfm["yield_bu_acre"].astype(float).values
                y_pred = dfm["y_pred"].astype(float).values
                mae, rmse, mape, r2, bias = compute_metrics(y_true, y_pred)
            else:
                mae = rmse = mape = r2 = bias = None
            return {
                "range_name": rname,
                "feature_season": feature_season,
                "model_name": model_name,
                "run_date": run_date,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2,
                "Bias": bias,
                "predictions_s3_key": key
            }

        if ranges:
            for r in ranges:
                name = r.get("name") or f"{r.get('start')}_{r.get('end')}"
                start_y = int(r.get("start"))
                end_y = int(r.get("end"))
                sub = merged[(merged["year"] >= start_y) & (merged["year"] <= end_y)]
                rows.append(_metric_row(name, sub))
        else:
            rows.append(_metric_row("all_years", merged))


    metrics = pd.DataFrame(rows)
    out_key = f"evaluation/run_date={RUN_DATE}/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/metrics.parquet"
    write_parquet_s3(metrics, out_key)
    print("Wrote metrics:", f"s3://{DATA_BUCKET}/{out_key}", flush=True)

    # Write a small ranking summary (best RMSE per range_name)
    if not metrics.empty:
        best = (
            metrics.dropna(subset=["RMSE"])
            .sort_values(["range_name","RMSE"])
            .groupby("range_name", as_index=False)
            .first()
        )
        best_key = f"evaluation/run_date={RUN_DATE}/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/_RANKING.json"
        s3.put_object(
            Bucket=DATA_BUCKET,
            Key=best_key,
            Body=json.dumps(best.to_dict(orient="records"))[:200000].encode("utf-8"),
            ContentType="application/json",
        )
        print("Wrote ranking:", f"s3://{DATA_BUCKET}/{best_key}", flush=True)


    if comp_rows:
        comp = pd.concat(comp_rows, ignore_index=True)
        comp_key = f"evaluation/run_date={RUN_DATE}/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/comparison.parquet"
        write_parquet_s3(comp, comp_key)
        print("Wrote comparison:", f"s3://{DATA_BUCKET}/{comp_key}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, flush=True)
        sys.exit(1)
