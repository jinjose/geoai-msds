import pandas as pd
import requests

def fetch_yield_year(api_key: str, year: int, state_fips: str, county_fips: str) -> pd.DataFrame:
    print(f"[YIELD] Fetching year={year} state={state_fips} county={county_fips}")
    base = "https://quickstats.nass.usda.gov/api/api_GET/"
    params = {
        "key": api_key,
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "commodity_desc": "CORN",
        "class_desc": "GRAIN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "agg_level_desc": "COUNTY",
        "year": str(year),
        "state_fips_code": str(state_fips).zfill(2),
        "format": "JSON",
    }
    if county_fips.upper() != "ALL":
        params["county_code"] = str(county_fips).zfill(3)

    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    rows = data.get("data", [])
    print(f"[YIELD] Raw rows returned: {len(rows)}")
    if not rows:
        print(f"[YIELD] No data for year={year}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"[YIELD] Processed rows: {len(df)}")

    if "Value" in df.columns:
        df["yield_bu_acre"] = df["Value"].astype(str).str.replace(",", "", regex=False)
        df["yield_bu_acre"] = pd.to_numeric(df["yield_bu_acre"], errors="coerce")

    keep = [c for c in ["year","state_fips_code","county_code","county_name","yield_bu_acre"] if c in df.columns]
    print(f"[YIELD] Final columns: {keep}")
    return df[keep].copy()

def ingest_yield(api_key: str, years_csv: str, state_fips: str, county_fips: str) -> pd.DataFrame:
    print(f"[YIELD] Years requested: {years}")
    years = [int(x.strip()) for x in years_csv.split(",") if x.strip()]
    frames = [fetch_yield_year(api_key, y, state_fips, county_fips) for y in years]
    print(f"[YIELD] Concatenating frames: {len(frames)}")
    print(f"[YIELD] Final DataFrame shape: {pd.concat(frames, ignore_index=True).shape[0] if frames else 0}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
