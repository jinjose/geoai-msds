"""Create SageMaker-compatible model.tar.gz for BYOC Batch Transform.

Example:
  python scripts/package_model_tar.py \
    --model-pkl path/to/model.pkl \
    --feature-list path/to/feature_list.txt \
    --out infra/model.tar.gz

Contents:
  model.pkl
  feature_list.txt  (optional)
  metadata.json     (optional)
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from datetime import datetime, timezone

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-pkl", required=True, help="Path to LightGBM Booster serialized via joblib to model.pkl")
    ap.add_argument("--feature-list", required=False, help="Optional newline-delimited feature list for contract checks")
    ap.add_argument("--metadata-json", required=False, help="Optional metadata.json path to include")

    ap.add_argument("--out", required=True, help="Output path for model.tar.gz")
    args = ap.parse_args()

    model_pkl = Path(args.model_pkl)
    if not model_pkl.exists():
        raise FileNotFoundError(model_pkl)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # auto metadata if not provided
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": "model.pkl",
    }
    tmp_meta = out.parent / "metadata.json"
    if args.metadata_json:
        meta_path = Path(args.metadata_json)
        meta_bytes = meta_path.read_bytes()
    else:
        meta_bytes = json.dumps(metadata, indent=2).encode("utf-8")

    # Additional files to include if present (same dir as model.pkl by default)
    extra_files = [
        (model_pkl.parent / "categorical_config.json", "categorical_config.json"),
        (model_pkl.parent / "feature_names.json", "feature_names.json"),
        (model_pkl.parent / "metrics_config.json", "metrics_config.json"),
    ]

    with tarfile.open(out, "w:gz") as tar:
        tar.add(model_pkl, arcname="model.pkl")

        if args.feature_list:
            fl = Path(args.feature_list)
            if not fl.exists():
                raise FileNotFoundError(fl)
            tar.add(fl, arcname="feature_list.txt")

        # Add extra files if they exist
        for file_path, arcname in extra_files:
            if file_path.exists():
                tar.add(file_path, arcname=arcname)

        tmp_meta.write_bytes(meta_bytes)
        tar.add(tmp_meta, arcname="metadata.json")

    tmp_meta.unlink(missing_ok=True)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
