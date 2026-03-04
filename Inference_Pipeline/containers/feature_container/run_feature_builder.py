import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # geoai_local
sys.path.insert(0, str(ROOT / "ingestion_container"))

from app.build_new_features import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, flush=True)
        sys.exit(1)
