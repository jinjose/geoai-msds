import argparse
from datetime import date
from pathlib import Path
import shutil

CUTOFFS = ["EARLY", "MID", "END"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", required=False, default=str(date.today()))
    ap.add_argument("--feature-dir", required=True, help="Folder containing features_EARLY.csv etc.")
    ap.add_argument("--out-dir", required=True, help="Base folder to write run_date=... folders")
    args = ap.parse_args()

    feature_dir = Path(args.feature_dir)
    out_base = Path(args.out_dir) / f"run_date={args.run_date}"
    out_base.mkdir(parents=True, exist_ok=True)

    for c in CUTOFFS:
        src = feature_dir / f"features_{c}.csv"
        if not src.exists():
            raise FileNotFoundError(f"Missing: {src}")
        dest_dir = out_base / f"cutoff={c}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest_dir / "features.csv")

    print(f"Prepared cutoff inputs at: {out_base}")

if __name__ == "__main__":
    main()