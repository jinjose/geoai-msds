import ee
import pandas as pd
from pathlib import Path
import time

# -----------------------------------------------------------
# Efficient ERA5-Land county aggregation (monthly loop + append)
# - Avoids holding all years in memory
# - Uses reduceRegions once per image (not per variable per county)
# - Writes ONE CSV per year
# -----------------------------------------------------------

ee.Initialize(project="msds-432-gcp-cloud")

START_YEAR = 2013
END_YEAR = 2024

STATE_NAME = "illinois"
STATE_ID = "17"  # Illinois = 17, Iowa = 19

# Counties for the chosen state
counties = ee.FeatureCollection("TIGER/2018/Counties")
state_counties = counties.filter(ee.Filter.eq("STATEFP", STATE_ID))

OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Friendly column names -> ERA5 band names
WEATHER_VARS = {
    "temperature": "temperature_2m",
    "rainfall": "total_precipitation_sum",
    "evapotranspiration": "total_evaporation_sum",
    "soil_moisture": "volumetric_soil_water_layer_1",
}

BANDS = list(WEATHER_VARS.values())
BAND_TO_FRIENDLY = {v: k for k, v in WEATHER_VARS.items()}


def _get_month_df(year: int, month: int, scale_m: int = 10000, retries: int = 3, sleep_s: int = 5) -> pd.DataFrame:
    """
    Download one month of daily county-mean weather.
    Returns a DataFrame for that month only.
    """
    print(f"  → {year}-{month:02d}")

    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .select(BANDS)
    )

    def per_image(img):
        # One server-side reduction per image across all counties for all bands
        reduced = img.reduceRegions(
            collection=state_counties,
            reducer=ee.Reducer.mean(),
            scale=scale_m,
            tileScale=4,
        )

        # Add metadata for each county feature row
        def add_meta(ft):
            return ft.set({
                "date": img.date().format("YYYY-MM-dd"),
                "year": year,
                "month": month,
                "county_fips": ft.get("GEOID"),
                "county_name": ft.get("NAME"),
            })

        return reduced.map(add_meta)

    fc = era5.map(per_image).flatten()

    # Retry getInfo() because EE can occasionally time out on large requests
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            features = fc.getInfo()
            records = [f["properties"] for f in features["features"]]
            df = pd.DataFrame(records)
            if df.empty:
                return df

            df["date"] = pd.to_datetime(df["date"])

            # Rename band columns to friendly names (temperature, rainfall, ...)
            rename_map = {band: BAND_TO_FRIENDLY.get(band, band) for band in BANDS}
            df = df.rename(columns=rename_map)

            return df
        except Exception as e:
            last_err = e
            print(f"    ⚠️  getInfo() failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download {year}-{month:02d} after {retries} attempts") from last_err


def extract_year_to_csv(year: int, out_dir: Path = OUT_DIR) -> Path:
    """
    Downloads one year and writes a single CSV file:
      era5_<state>_county_<year>.csv

    It appends month by month to avoid memory blow-ups.
    """
    year_path = out_dir / f"era5_{STATE_NAME}_county_{year}.csv"
    if year_path.exists():
        year_path.unlink()  # overwrite cleanly

    wrote_header = False

    print(f"\nExtracting {STATE_NAME.upper()} ERA5 for {year} (monthly chunks)")
    for month in range(1, 13):
        df = _get_month_df(year, month)

        # Append to per-year CSV
        if not df.empty:
            df.to_csv(year_path, mode="a", index=False, header=not wrote_header)
            wrote_header = True

        # Free memory ASAP
        del df

    print(f"  ✔ Saved: {year_path}")
    return year_path


def extract_multiple_years(start_year: int, end_year: int):
    """
    Writes one file per year, and also an optional all-years file created by concatenation on disk.
    """
    per_year_files = []
    for year in range(start_year, end_year + 1):
        per_year_files.append(extract_year_to_csv(year))

    # Optional: create an all-years file WITHOUT holding all dataframes in memory
    all_years_path = OUT_DIR / f"era5_{STATE_NAME}_county_all_years.csv"
    if all_years_path.exists():
        all_years_path.unlink()

    wrote_header = False
    for p in per_year_files:
        chunk = pd.read_csv(p)
        chunk.to_csv(all_years_path, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        del chunk

    print("\n✔ County-wise ERA5 extraction complete")
    print(f"Per-year files: {len(per_year_files)}")
    print(f"All-years file: {all_years_path}")


if __name__ == "__main__":
    extract_multiple_years(START_YEAR, END_YEAR)
