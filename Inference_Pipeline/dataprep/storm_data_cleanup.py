import argparse
import pandas as pd
from pathlib import Path

DEFAULT_EVENT_TYPES = ["Thunderstorm Wind", "High Wind"]

def _norm_county(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace(" county", "").strip()
    return s

def clean_storm_events_to_daily(
    input_csv: str,
    output_path: str,
    wind_cutoff_mph: float = 58.0,
    state_name: str = "IOWA",
    event_types=None,
) -> pd.DataFrame:
    """
    Input: NOAA Storm Events CSV (event-level).
    Output: daily county-level table with max wind per county-date.
    """
    event_types = event_types or DEFAULT_EVENT_TYPES

    df = pd.read_csv(input_csv)

    # --- Basic required columns check (based on your sample) ---
    required = ["STATE", "EVENT_TYPE", "BEGIN_DATE_TIME", "MAGNITUDE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # --- Filter to Iowa ---
    df = df[df["STATE"].astype(str).str.upper() == state_name.upper()].copy()

    # --- Filter to wind-related events (keep list configurable) ---
    df["EVENT_TYPE"] = df["EVENT_TYPE"].astype(str)
    df = df[df["EVENT_TYPE"].isin(event_types)].copy()

    # --- Parse datetime ---
    df["datetime"] = pd.to_datetime(df["BEGIN_DATE_TIME"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # --- Magnitude to numeric (wind mph) ---
    df["wind_mph"] = pd.to_numeric(df["MAGNITUDE"], errors="coerce")
    df = df.dropna(subset=["wind_mph"])

    # Optional: sanity bounds (avoid corrupt values)
    df = df[(df["wind_mph"] >= 0) & (df["wind_mph"] <= 200)]

    # --- County name: prefer CZ_NAME if present; else use 'county' column from your file ---
    if "CZ_NAME" in df.columns:
        df["county_norm"] = df["CZ_NAME"].apply(_norm_county)
    elif "county" in df.columns:
        df["county_norm"] = df["county"].apply(_norm_county)
    else:
        raise ValueError("No CZ_NAME or county column found to derive county.")

    df = df[df["county_norm"] != ""]

    # --- Convert to daily (date only) ---
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year

    # --- IMPORTANT: dedup/aggregate to daily county max wind ---
    daily = (
        df.groupby(["county_norm", "date", "year"], as_index=False)
          .agg(
              wind_mph=("wind_mph", "max"),
              event_count=("wind_mph", "size"),
          )
    )

    # --- Severe flag (your model cutoff) ---
    daily["severe_gust_58"] = (daily["wind_mph"] >= float(wind_cutoff_mph)).astype(int)

    # Rename county column to match your pipeline naming
    daily = daily.rename(columns={"county_norm": "county"})

    # --- Save output ---
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() in [".parquet"]:
        daily.to_parquet(out, index=False)
    else:
        daily.to_csv(out, index=False)

    return daily

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output", required=True, help="Output .parquet or .csv")
    ap.add_argument("--wind_cutoff_mph", type=float, default=58.0)
    ap.add_argument("--state", default="IOWA")
    ap.add_argument("--event_types", default="Thunderstorm Wind,High Wind",
                    help="Comma-separated list of EVENT_TYPE values to keep")
    args = ap.parse_args()

    event_types = [x.strip() for x in args.event_types.split(",") if x.strip()]
    daily = clean_storm_events_to_daily(
        input_csv=args.input_csv,
        output_path=args.output,
        wind_cutoff_mph=args.wind_cutoff_mph,
        state_name=args.state,
        event_types=event_types,
    )
    print(f"Rows written: {len(daily)}")
    print(daily.head(10).to_string(index=False))

if __name__ == "__main__":
    main()