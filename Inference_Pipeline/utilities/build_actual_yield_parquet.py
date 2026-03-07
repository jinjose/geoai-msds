import re
from pathlib import Path
import pandas as pd

def normalize_county_name(x: str) -> str:
    s = str(x).strip().lower()
    s = s.replace(" county", "")
    s = " ".join(s.split())
    return s

def build_actuals_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # --- Filter to Corn grain yield in BU/ACRE ---
    # Your file has multiple "Data Item" types (grain yield, silage yield, etc.)
    df["Data Item"] = df["Data Item"].astype(str)

    grain_mask = df["Data Item"].str.contains(
        r"corn.*grain.*yield.*bu\s*/\s*acre",
        flags=re.IGNORECASE,
        regex=True,
        na=False,
    )

    df = df[grain_mask].copy()

    # Optional extra filters (usually safe)
    if "Period" in df.columns:
        df = df[df["Period"].astype(str).str.upper().eq("YEAR")].copy()

    # --- Select and rename columns ---
    # County column is "County", Year column is "Year", yield is "Value"
    out = df[["County", "Year", "Value"]].copy()
    out.columns = ["county", "year", "yield_bu_acre"]

    # --- Clean ---
    out["county"] = out["county"].astype(str).map(normalize_county_name)
    out = out[out["county"].notna() & (out["county"] != "")].copy()

    # Remove USDA aggregate bucket rows
    out = out[~out["county"].isin(["other counties", "other"])].copy()

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["yield_bu_acre"] = pd.to_numeric(out["yield_bu_acre"], errors="coerce")

    out = out.dropna(subset=["year", "yield_bu_acre"]).copy()

    # --- Deduplicate to 1 row per county-year ---
    # If duplicates exist (rare), keep the first non-null
    out = (
        out.sort_values(["county", "year"])
           .drop_duplicates(subset=["county", "year"], keep="first")
           .reset_index(drop=True)
    )

    return out

def main():
    # Change these paths as needed
    csv_path = r"D:/MS_DataScience/MSDS_498_Capstone_Final/GIT/IOWA_county_wise_yield.csv"   # <-- your input
    out_path = Path("actuals.parquet")

    actuals = build_actuals_from_csv(csv_path)

    print("Columns:", actuals.columns.tolist())
    print("Year range:", int(actuals["year"].min()), int(actuals["year"].max()))
    print("Rows:", len(actuals), "| counties:", actuals["county"].nunique())
    print(actuals.head())

    actuals.to_parquet(out_path, index=False, engine="pyarrow")
    print("Wrote:", out_path.resolve())

    # OPTIONAL: upload to S3 (requires AWS credentials configured)
    # Example target:
    # s3://geoai-demo-data/curated/yield/state_fips=19/county_fips=ALL/actuals.parquet
    try:
        import boto3
        bucket = "geoai-demo-data"
        state_fips = "19"
        key = f"curated/yield/state_fips={state_fips}/county_fips=ALL/actuals.parquet"

        s3 = boto3.client("s3")
        s3.upload_file(str(out_path), bucket, key)
        print("Uploaded to:", f"s3://{bucket}/{key}")
    except Exception as e:
        print("Skipped S3 upload (ok). Reason:", repr(e))

if __name__ == "__main__":
    main()