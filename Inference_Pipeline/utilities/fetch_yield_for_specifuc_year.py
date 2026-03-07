import pandas as pd
import requests


def fetch_yield_2020(api_key: str, state_fips: str = "19", county_fips: str = "ALL") -> pd.DataFrame:
    year = 2020
    print(f"[YIELD] Fetching year={year} state={state_fips} county={county_fips}")

    base = "https://quickstats.nass.usda.gov/api/api_GET/"

    # Minimal, exact query
    params = {
        "key": api_key,
        "source_desc": "SURVEY",
        "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
        "domain_desc": "TOTAL",
        "agg_level_desc": "COUNTY",
        "year": str(year),
        "state_fips_code": str(state_fips).zfill(2),
        "format": "JSON",
    }

    if str(county_fips).upper() != "ALL":
        params["county_code"] = str(county_fips).zfill(3)

    r = requests.get(base, params=params, timeout=60)

    print("Request URL:", r.url)
    print("Status:", r.status_code)
    print("Response text:", r.text[:2000])

    r.raise_for_status()

    data = r.json()
    rows = data.get("data", [])
    print(f"[YIELD] Raw rows returned: {len(rows)}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "Value" in df.columns:
        df["yield_bu_acre"] = (
            df["Value"].astype(str).str.replace(",", "", regex=False)
        )
        df["yield_bu_acre"] = pd.to_numeric(df["yield_bu_acre"], errors="coerce")

    keep = [
        c for c in [
            "year",
            "state_name",
            "state_fips_code",
            "county_name",
            "county_code",
            "short_desc",
            "domain_desc",
            "Value",
            "yield_bu_acre",
        ]
        if c in df.columns
    ]

    out = df[keep].copy()
    print(f"[YIELD] Final rows: {len(out)}")
    print(f"[YIELD] Final columns: {list(out.columns)}")
    expected_counties = 99
    actual_counties = out["county_name"].nunique()

    print(f"[YIELD] Unique counties returned: {actual_counties}")

    if actual_counties != expected_counties:
        print(f"[YIELD][WARN] Expected {expected_counties} Iowa counties, got {actual_counties}")

    return out


if __name__ == "__main__":
    API_KEY = "EB1AC2E1-A44D-3D7A-93E4-09E6D8B6ADAC"
    STATE_FIPS = "19"
    COUNTY_FIPS = "ALL"

    df = fetch_yield_2020(API_KEY, STATE_FIPS, COUNTY_FIPS)

    if not df.empty:
        out_file = "corn_yield_iowa_county_2020.csv"
        df.to_csv(out_file, index=False)
        print(f"[YIELD] Saved to {out_file}")
    else:
        print("[YIELD] No data returned")