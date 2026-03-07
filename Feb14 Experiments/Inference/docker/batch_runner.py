
"""Entry point for SageMaker Batch Transform BYOC container.

SageMaker mounts:
  - /opt/ml/input/data/<channel>/...  (input files)
  - /opt/ml/model/                   (model artifacts extracted from model.tar.gz)
  - /opt/ml/output/                  (write outputs here)

This script:
  1) Locates the first CSV file under the 'input' channel.
  2) Runs LightGBM inference.
  3) Writes a CSV to /opt/ml/output/predictions.csv
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from time import time

from geoai.inference.predict import predict_csv

INPUT_CHANNEL = os.environ.get("SM_INPUT_CHANNEL", "input")
INPUT_DIR = Path(f"/opt/ml/input/data/{INPUT_CHANNEL}")
OUTPUT_DIR = Path("/opt/ml/output")
OUTPUT_FILE = OUTPUT_DIR / "predictions.csv"

# Optional: columns to exclude from features
ID_COLS = [c.strip() for c in os.environ.get("ID_COLUMNS", "county,year,cutoff").split(",") if c.strip()]

# Optional feature contract file packaged in model.tar.gz
FEATURE_LIST = Path(os.environ.get("FEATURE_LIST_PATH", "/opt/ml/model/feature_list.txt"))

# def _find_input_csv() -> Path:
#     candidates = sorted(INPUT_DIR.glob("**/*.csv"))
#     if not candidates:
#         raise FileNotFoundError(f"No CSV files found under {INPUT_DIR}")
#     return candidates[0]

def _find_input_csv() -> Path:
   
    # print("Input directory structure:")
    # for p in Path("/opt/ml/input/data").rglob("*"):
    #     print(p)

    # base = Path("/opt/ml/input/data")
    # candidates = sorted(base.glob("**/*.csv"))
    # candidates = [p for p in candidates if p.is_file()]
    # if not candidates:
    #     raise FileNotFoundError(f"No CSV files found under {base}. Check TransformInput S3Uri.")
    # return candidates[0]/
    roots = [Path("/opt/ml/input/data"), Path(f"/opt/ml/input/data/{INPUT_CHANNEL}")]
    for r in roots:
        if r.exists():
            candidates = sorted([p for p in r.rglob("*.csv") if p.is_file()])
            if candidates:
                return candidates[0]
    raise FileNotFoundError("No CSV files found under /opt/ml/input/data or channel dir. Check TransformInput S3Uri.")


def main() -> None:
    input_csv = _find_input_csv()
    print(f"[batch_runner] input_csv={input_csv}")
    print(f"[batch_runner] output_csv={OUTPUT_FILE}")
    res = predict_csv(
        input_csv=input_csv,
        output_csv=OUTPUT_FILE,
        id_columns=ID_COLS,
        expected_feature_list_path=FEATURE_LIST if FEATURE_LIST.exists() else None,
    )
    print(f"[batch_runner] model_path={res.model_path}")
    print(f"[batch_runner] num_rows={len(res.df_out)} num_features={len(res.feature_columns)}")
    print("[batch_runner] done")

if __name__ == "__main__":
    main()
