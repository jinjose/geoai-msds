import os
import sys
import pandas as pd
import boto3
import pyarrow.dataset as ds
from pathlib import Path
import tempfile
import pyarrow as pa
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]   # geoai_local
sys.path.insert(0, str(ROOT / "feature_container"))

from app.config import FORECAST_CUTOFFS, load_expected_features_from_schema
from app.preprocessing.cleaning import clean_yield, clean_ndvi, clean_weather, smooth_county_ndvi, clean_storm_partitioned,apply_ndvi_cutoff_fallback
from app.features.build_features import build_feature_table

DATA_BUCKET = os.environ["DATA_BUCKET"]
AWS_REGION = os.environ.get("AWS_REGION","ap-south-1")

STATE_FIPS = os.environ["STATE_FIPS"]
COUNTY_FIPS = os.environ["COUNTY_FIPS"]  # 'ALL' allowed
RUN_DATE = os.environ["RUN_DATE"]
FEATURE_SEASON = os.environ["FEATURE_SEASON"]
# Export years: for inference we typically export the RUN_DATE year.
# Optionally override with EXPORT_YEARS="2025,2026"
# Forecast (inference) year you want features for
PREDICT_YEAR = int(os.environ["PREDICT_YEAR"])

# Baseline years for "historical mean/anomaly" features (NDVI/ERA5)
# Default to BACKFILL_START_YEAR if provided; otherwise 2014.
BASELINE_START_YEAR = int(os.environ.get("BASELINE_START_YEAR", os.environ.get("BACKFILL_START_YEAR", "2014")))

# Yield labels exist only up to this year (training history)
YIELD_END_YEAR = int(os.environ.get("YIELD_END_YEAR", str(PREDICT_YEAR - 1)))

# Export years: for inference we export ONLY PREDICT_YEAR unless explicitly overridden

EXPORT_YEARS = os.environ.get("EXPORT_YEARS", "")
if EXPORT_YEARS.strip():
    export_years = [int(x.strip()) for x in EXPORT_YEARS.split(",") if x.strip()]
else:
    # For inference we want frozen features for the predict year (e.g., 2025)
    export_years = [PREDICT_YEAR]

GRANULARITY = os.environ.get("GRANULARITY", "daily")  # or "daily" if you have daily features/labels
COMMODITY = os.environ.get("COMMODITY", "corn")  # or "soybean" if you have multiple commodities and want to partition by that as well


s3 = boto3.client("s3", region_name=AWS_REGION)

def normalize_county(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(" county", "", regex=False)
    )

s3 = boto3.client("s3")

def _download_s3_to_tmp(bucket: str, key: str) -> Path:
    fd, local_path = tempfile.mkstemp(suffix=os.path.basename(key).replace("/", "_"))
    os.close(fd)
    s3.download_file(bucket, key, local_path)
    return local_path
    # tmpdir = Path(tempfile.mkdtemp())
    # local = tmpdir / Path(key).name
    # s3.download_file(bucket, key, str(local))
    # return local

def read_raw_dataset(dataset_name: str) -> pd.DataFrame:
    import os
    import pandas as pd

    # ---------- dataset-specific granularity ----------
    if dataset_name == "era5":
        granularity = os.getenv("ERA5_GRANULARITY", "daily")
    elif dataset_name == "ndvi":
        granularity = os.getenv("NDVI_GRANULARITY", "daily")
    elif dataset_name == "storm":
        granularity = os.getenv("STORM_GRANULARITY", "daily")
    elif dataset_name == "yield":
        granularity = os.getenv("YIELD_GRANULARITY", "yearly")
    else:
        granularity = os.getenv("GRANULARITY", "daily")

    base_prefix = (
        f"raw/dataset={dataset_name}/"
        f"state_fips={STATE_FIPS}/"
        f"county_fips={COUNTY_FIPS}/"
        f"granularity={granularity}/"
    )

    predict_year = int(os.getenv("PREDICT_YEAR", "2025"))
    target_years_env = os.getenv("TARGET_YEARS", "").strip()

    if dataset_name == "yield":
        years_to_load = list(range(2013, YIELD_END_YEAR + 1))
    else:
        if target_years_env:
            years_to_load = [int(y.strip()) for y in target_years_env.split(",") if y.strip()]
        else:
            # For NDVI/ERA5 we load a historical baseline window for anomaly features.
            if dataset_name in ("ndvi", "era5"):
                years_to_load = list(range(BASELINE_START_YEAR, predict_year + 1))
            else:
                years_to_load = [predict_year]

    # ---------- list keys under a prefix ----------
    def list_keys(prefix: str) -> list[str]:
        keys = []
        token = None
        while True:
            kwargs = {"Bucket": DATA_BUCKET, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return keys

    parts = []

    for y in years_to_load:
        year_prefix = base_prefix + f"year={y}/"
        keys = list_keys(year_prefix)

        parquet_keys = [k for k in keys if k.lower().endswith(".parquet")]
        csv_keys = [k for k in keys if k.lower().endswith(".csv")]

        # =========================================================
        # 1) Parquet path: READ FILE-BY-FILE (no schema merge ever)
        # =========================================================
        if parquet_keys:
            dfs = []
            for pk in parquet_keys:
                try:
                    local = _download_s3_to_tmp(DATA_BUCKET, pk)
                    dfi = pd.read_parquet(local)

                    # normalize partition cols if present
                    for c in ["year", "month", "day"]:
                        if c in dfi.columns:
                            dfi[c] = pd.to_numeric(dfi[c], errors="coerce").astype("Int64")

                    # Yield: ALWAYS derive year from 'Year' if present (source of truth)
                    if dataset_name == "yield" and "Year" in dfi.columns:
                        dfi["year"] = pd.to_numeric(dfi["Year"], errors="coerce").astype("Int64")

                    dfs.append(dfi)
                except Exception as e:
                    print(f"[{dataset_name}] parquet file read failed: {pk} -> {e}")

            if dfs:
                parts.append(pd.concat(dfs, ignore_index=True))
                continue  # do not also read CSVs (avoid duplicates)

        # =========================================================
        # 2) CSV fallback
        # =========================================================
        for ck in csv_keys:
            local = _download_s3_to_tmp(DATA_BUCKET, ck)
            df = pd.read_csv(local)

            # Yield: prefer 'Year' if present
            if dataset_name == "yield" and "Year" in df.columns:
                df["year"] = pd.to_numeric(df["Year"], errors="coerce")
            elif "year" not in df.columns:
                df["year"] = int(y)

            parts.append(df)

    if not parts:
        print(f"[{dataset_name}] No files found for prefix={base_prefix}")
        return pd.DataFrame()

    df_out = pd.concat(parts, ignore_index=True)

    # final normalize year
    if dataset_name == "yield" and "Year" in df_out.columns:
        df_out["year"] = pd.to_numeric(df_out["Year"], errors="coerce").astype("Int64")
    elif "year" in df_out.columns:
        df_out["year"] = pd.to_numeric(df_out["year"], errors="coerce").astype("Int64")

    print(f"[{dataset_name}] granularity={granularity}, years={years_to_load}")
    print(f"[{dataset_name}] rows={len(df_out)}")
    if "year" in df_out.columns:
        u = df_out["year"].dropna().unique().tolist()
        u_sorted = sorted([int(x) for x in u]) if u else []
        print(f"[{dataset_name}] year uniques={u_sorted[:15]}")
    else:
        print(f"[{dataset_name}] WARNING: no 'year' column after load. cols={list(df_out.columns)}")

    return df_out

def put_parquet(df: pd.DataFrame, s3_key: str):
    # write local parquet then upload
    tmp = Path(tempfile.mkdtemp()) / "data.parquet"
    df.to_parquet(tmp, index=False)
    s3.upload_file(str(tmp), DATA_BUCKET, s3_key)

def canon_county(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.replace(" county", "", regex=False)
         .str.replace(r"\s+", " ", regex=True)
    )

def ensure_county_col(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = df.copy()

    # pick the right source column with minimal assumptions
    if "county" in df.columns:
        src = "county"
    elif "county_name" in df.columns:
        src = "county_name"
    elif "County" in df.columns:
        src = "County"
    else:
        raise ValueError(f"{dataset}: no county column found. cols={list(df.columns)}")

    df["county"] = canon_county(df[src])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["county", "year"])
    return df

def main():
    # Read all counties for the requested state/county from partitioned lake.
    # Your ingest_raw.py writes state_fips and county_fips columns; we filter in pandas for simplicity.
    yield_raw = read_raw_dataset("yield")
    ndvi_raw  = read_raw_dataset("ndvi")
    wx_raw    = read_raw_dataset("era5")
    storm_raw = read_raw_dataset("storm")  # may be empty if not ingested

    print("RAW columns:", list(yield_raw.columns))
    print("RAW row count:", len(yield_raw))

    # If the raw file has "Year" column (USDA CSV style)
    if "Year" in yield_raw.columns:
        years = pd.to_numeric(yield_raw["Year"], errors="coerce")
        print("RAW years (unique):", sorted(years.dropna().unique()))
        print("RAW 2024 rows:", (years == 2024).sum())
    else:
        # if already standardized somewhere
        if "year" in yield_raw.columns:
            years = pd.to_numeric(yield_raw["year"], errors="coerce")
            print("RAW years (unique):", sorted(years.dropna().unique()))
            print("RAW 2024 rows:", (years == 2024).sum())
    

    # Filter requested geography
    #yield_raw = yield_raw[yield_raw["state_fips"].astype(str) == str(STATE_FIPS)]
    #ndvi_raw  = ndvi_raw[ndvi_raw["state_fips"].astype(str) == str(STATE_FIPS)]
    #wx_raw    = wx_raw[wx_raw["state_fips"].astype(str) == str(STATE_FIPS)]
    #if not storm_raw.empty:
    #    storm_raw = storm_raw[storm_raw["state_fips"].astype(str) == str(STATE_FIPS)]

    if COUNTY_FIPS != "ALL":
        for name, df in [("yield", yield_raw), ("ndvi", ndvi_raw), ("era5", wx_raw), ("storm", storm_raw)]:
            if not df.empty and "county_fips" in df.columns:
                df = df[df["county_fips"].astype(str) == str(COUNTY_FIPS)]
            # re-assign back
            if name == "yield": yield_raw = df
            elif name == "ndvi": ndvi_raw = df
            elif name == "era5": wx_raw = df
            else: storm_raw = df

    # Clean expects columns used in your original logic
    yield_df = clean_yield(yield_raw)

    yield_df = ensure_county_col(yield_df, "yield")

    print("CLEANED years:", sorted(yield_df["year"].unique()))
    #print("CLEANED 2024 rows:", (yield_df["year"] == 2024).sum())
    # keep only labeled yield history (exclude predict year)
    yield_df = yield_df[yield_df["year"].astype(int) <= YIELD_END_YEAR].copy()

    ndvi = clean_ndvi(ndvi_raw)
    ndvi  = ensure_county_col(ndvi,  "ndvi")
    print("wx_raw columns:", wx_raw.columns.tolist())
    print("wx_raw sample rows:", wx_raw.head(2))
    wx = clean_weather(wx_raw)
    wx = ensure_county_col(wx, "era5")
    storm_df = clean_storm_partitioned(storm_raw) if not storm_raw.empty else pd.DataFrame()
    if not storm_df.empty:
        storm_df = ensure_county_col(storm_df, "storm")

    for d in (ndvi, wx):
        if "year" in d.columns:
            d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64").astype(int)

    if not storm_df.empty and "year" in storm_df.columns:
        storm_df["year"] = pd.to_numeric(storm_df["year"], errors="coerce").astype("Int64").astype(int)
    else:
        print("No storm data available after cleaning.")
    
    # Normalize county labels
    yield_df["county"] = normalize_county(yield_df["county"])
    ndvi["county"] = normalize_county(ndvi["county"])
    wx["county"] = normalize_county(wx["county"])
    if not storm_df.empty:
        storm_df["county"] = normalize_county(storm_df["county"])

    ndvi = apply_ndvi_cutoff_fallback(ndvi, predict_year=PREDICT_YEAR, feature_season=FEATURE_SEASON)
    # Smooth NDVI
    ndvi = smooth_county_ndvi(ndvi, window=9, poly=2)

    # Align NDVI + WX
    common_counties = set(ndvi["county"]).intersection(set(wx["county"]))
    common_years = set(ndvi["year"]).intersection(set(wx["year"]))

    ndvi = ndvi[(ndvi["county"].isin(common_counties)) & (ndvi["year"].isin(common_years))]
    wx = wx[(wx["county"].isin(common_counties)) & (wx["year"].isin(common_years))]

        # "Historical" baselines should exclude the prediction year itself to avoid mild leakage
    ndvi_base = ndvi[ndvi["year"] < PREDICT_YEAR]
    wx_base = wx[wx["year"] < PREDICT_YEAR]
    ndvi_hist_mean = ndvi_base.groupby("county")["NDVI"].mean()
    temp_hist_mean = wx_base.groupby("county")["temperature"].mean()

    # Cutoff
    month, day = FORECAST_CUTOFFS[FEATURE_SEASON]

    print("Storm rows:", 0 if storm_df is None else len(storm_df), "cols:", None if storm_df is None else list(storm_df.columns))

    feature_df = build_feature_table(
        yield_df, ndvi, wx,
        month, day,
        ndvi_hist_mean, temp_hist_mean,
        target_years=[PREDICT_YEAR],
        storm_df=storm_df
    )
    if feature_df.empty:
        raise RuntimeError("Feature table is empty. Check ingestion outputs / cutoff.")
    
    # # Actuals parquet (for eval)
    # actuals = feature_df[["county","year","yield_bu_acre"]].copy()
    # actual_key = f"curated/yield/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/actuals.parquet"
    # put_parquet(actuals, actual_key)
    # --- Actuals parquet (historical labels for evaluation/backtests) ---
    # Write actual observed yields (2013..YIELD_END_YEAR) from cleaned yield_df
    actuals = (
        yield_df[["county", "year", "yield_bu_acre"]]
        .copy()
    )
    # keep only valid label years + non-null yields
    actuals["year"] = pd.to_numeric(actuals["year"], errors="coerce")
    actuals = actuals.dropna(subset=["year", "yield_bu_acre"])
    actuals["year"] = actuals["year"].astype(int)

    actuals = actuals[
        (actuals["year"] >= 2013) &
        (actuals["year"] <= YIELD_END_YEAR)
    ].copy()

    actual_key = f"curated/yield/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/granularity={GRANULARITY}/year={PREDICT_YEAR}/actuals.parquet"  
    put_parquet(actuals, actual_key)
    print(f"Actuals rows written: {len(actuals)} years={sorted(actuals['year'].unique())[:3]}..{sorted(actuals['year'].unique())[-3:]}", flush=True)

    # Frozen features parquet (NO label)
    feature_df = feature_df.drop(columns=["yield_bu_acre"])
    # 3) load expected features from schema (source of truth)
    expected = load_expected_features_from_schema(FEATURE_SEASON, schema_path=os.getenv("FEATURE_SCHEMA_PATH"))
    # Select only features used by this season/model (avoids extra columns)
    selected = ["year"] + expected
    # dedupe
    seen=set(); selected=[c for c in selected if not (c in seen or seen.add(c))]
    missing = [c for c in selected if c not in feature_df.columns]

    if missing:
        raise RuntimeError(
            f"Frozen feature mismatch: missing {len(missing)} expected columns for season={FEATURE_SEASON}. "
            f"First 20 missing: {missing[:20]}. "
            f"Available columns sample: {sorted(feature_df.columns)[:40]}"
        )    
    
    frozen_feature_df = feature_df.loc[feature_df["year"].isin(export_years), selected].copy()

    required = ["rolling_3yr_mean","ndvi_peak","ndvi_slope","temp_anomaly","net_moisture_stress","heat_days_gt32","wind_severe_days_58_cutoff"]
    bad = frozen_feature_df[required].isna().any(axis=1)
    if bad.any():
        print("Dropping rows with missing required features:", int(bad.sum()))
        frozen_feature_df = frozen_feature_df.loc[~bad].copy()

    # IMPORTANT: match Step Functions TransformInput prefix
    frozen_key = (
        f"features_frozen/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/"
        f"predict_year={PREDICT_YEAR}/feature_season={FEATURE_SEASON}/run_date={RUN_DATE}/part.parquet"
    )
    put_parquet(frozen_feature_df, frozen_key)

    print("FREEZE OUTPUT ROWS:", len(frozen_feature_df))
    print("FREEZE OUTPUT COLS:", len(frozen_feature_df.columns))
    print("FREEZE OUTPUT COL NAMES:", sorted(frozen_feature_df.columns)[:50])

    print("Wrote:", f"s3://{DATA_BUCKET}/{actual_key}", flush=True)
    print("Wrote:", f"s3://{DATA_BUCKET}/{frozen_key}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, flush=True)
        sys.exit(1)
