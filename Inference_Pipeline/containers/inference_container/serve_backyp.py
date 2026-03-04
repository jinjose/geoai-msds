import os
import sys
import json
import pandas as pd
import numpy as np
import mlflow.pyfunc

MODEL_DIR = "/opt/ml/model"
INPUT_DIR = "/opt/ml/input/data"
OUTPUT_DIR = "/opt/ml/output/data"

print("SERVE VERSION:", __file__, flush=True)


def find_input_files():
    files = []
    for root, _, fs in os.walk(INPUT_DIR):
        for f in fs:
            if f.lower().endswith(".parquet"):
                files.append(os.path.join(root, f))
    if not files:
        # Helpful debug: show what actually exists
        for root, _, fs in os.walk(INPUT_DIR):
            for f in fs:
                print("FOUND NON-PARQUET:", os.path.join(root, f), flush=True)
        raise RuntimeError(f"No parquet file found under {INPUT_DIR}")
    return sorted(files)


def write_output(df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "predictions.parquet")
    df.to_parquet(out_path, index=False)
    print("Wrote:", out_path, flush=True)


def load_feature_schema() -> dict:
    schema_path = os.path.join(MODEL_DIR, "feature_schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError("feature_schema.json not found in model directory")
    with open(schema_path, "r") as f:
        return json.load(f)


def prepare_inference_dataframe(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    expected_features = schema["expected_features"]
    categorical_features = schema.get("categorical_features", [])
    target_col = schema.get("target")

    dfX = df.copy()

    if target_col and target_col in dfX.columns:
        dfX = dfX.drop(columns=[target_col])

    for col in expected_features:
        if col not in dfX.columns:
            dfX[col] = np.nan

    dfX = dfX[expected_features]

    for col in expected_features:
        if col in categorical_features:
            dfX[col] = dfX[col].astype("category")
        else:
            dfX[col] = pd.to_numeric(dfX[col], errors="coerce")

    return dfX


def main():
    print("Starting batch inference...", flush=True)

    paths = find_input_files()
    print("Found parquet files:", len(paths), flush=True)
    for p in paths[:10]:
        print("  parquet:", p, flush=True)

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    if df.empty:
        raise RuntimeError("Input parquet(s) are empty after concatenation")

    # Normalize county names
    if "county" in df.columns:
        df["county"] = df["county"].astype(str).str.strip().str.lower()

        # Remove accidental statewide rows
        df = df[~df["county"].isin(["iowa", "statewide", "all"])]

    # Drop any ground-truth columns
    for col in ["yield_bu_acre", "y_true"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if "year" not in df.columns:
        raise RuntimeError("Input parquet must contain 'year' column")

    # Load model + schema
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    schema = load_feature_schema()

    # Align features
    dfX = prepare_inference_dataframe(df, schema)
    print("Feature matrix shape:", dfX.shape, flush=True)

    # Predict (ensure 1-D)
    y_pred = np.asarray(model.predict(dfX)).reshape(-1)

    # Build output
    id_cols = [c for c in ["county", "year"] if c in df.columns]
    out = df[id_cols].copy()
    out["prediction"] = y_pred.astype("float64")

    # Aggregate safety
    out = (
        out.groupby(["county", "year"], as_index=False)
           .agg(prediction=("prediction", "mean"))
    )

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype("int64")

    print("Predictions generated:", len(out), flush=True)

    write_output(out)
    print("Batch inference completed successfully.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[SERVE] ERROR:", e, flush=True)
        sys.exit(1)