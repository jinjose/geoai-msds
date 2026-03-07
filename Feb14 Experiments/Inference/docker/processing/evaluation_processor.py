import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

INPUT_DIR = Path("/opt/ml/processing/input")
PRED_DIR = Path("/opt/ml/processing/predictions")
OUT_DIR = Path("/opt/ml/processing/output")

TARGET_COL = os.getenv("TARGET_COL", "yield_bu_acre")
CUTOFF = os.getenv("CUTOFF", "UNKNOWN")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = list(INPUT_DIR.glob("*.csv"))
    pred_files = list(PRED_DIR.glob("*.csv"))

    if not input_files:
        raise RuntimeError("No input CSV found")

    if not pred_files:
        raise RuntimeError("No prediction CSV found")

    df = pd.read_csv(input_files[0])
    preds = pd.read_csv(pred_files[0], header=None)
    preds.columns = ["y_pred"]

    df = df.reset_index(drop=True)
    preds = preds.reset_index(drop=True)
    df["y_pred"] = preds["y_pred"]

    metrics = {}

    if TARGET_COL in df.columns:
        y_true = df[TARGET_COL].values
        y_pred = df["y_pred"].values

        metrics = {
            "cutoff": CUTOFF,
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "n": int(len(y_true))
        }

    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    df.to_parquet(OUT_DIR / "scored.parquet", index=False)

    print("Evaluation complete")

if __name__ == "__main__":
    main()