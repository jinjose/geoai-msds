import ee
import pandas as pd
from pathlib import Path
import argparse

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
PROJECT_ID = "XXXXXX"

HIST_START_YEAR = 2012
HIST_END_YEAR = 2024
LIVE_YEAR = 2025

START_MONTH = 4   # April
END_MONTH = 9     # September

FINAL_FILENAME = "era5_iowa_state.csv"

HISTORICAL_OUT_DIR = Path("training/raw")
LIVE_OUT_DIR = Path("inference/raw")

# -----------------------------------------------------------
# Initialize Earth Engine
# -----------------------------------------------------------
ee.Initialize(project=PROJECT_ID)

# -----------------------------------------------------------
# Iowa counties
# -----------------------------------------------------------
counties = ee.FeatureCollection("TIGER/2018/Counties")
iowa_counties = counties.filter(ee.Filter.eq("STATEFP", "19"))

# -----------------------------------------------------------
# ERA5 Variables
# -----------------------------------------------------------
weather_vars = {
    "temperature": "temperature_2m",
    "rainfall": "total_precipitation_sum",
    "evapotranspiration": "total_evaporation_sum",
    "soil_moisture": "volumetric_soil_water_layer_1"
}

# -----------------------------------------------------------
# Extract weather for one month
# -----------------------------------------------------------
def extract_weather_for_year_month(year: int, month: int) -> pd.DataFrame:

    print(f"  → {year}-{month:02d}")

    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
    )

    def per_image(image):

        def per_county(county):

            props = {
                "date": image.date().format("YYYY-MM-dd"),
                "year": year,
                "month": month,
                "county_fips": county.get("GEOID"),
                "county_name": county.get("NAME"),
            }

            for name, band in weather_vars.items():
                val = image.select(band).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=county.geometry(),
                    scale=10000,
                    bestEffort=True,
                    maxPixels=1e13
                ).get(band)

                props[name] = val

            return ee.Feature(None, props)

        return iowa_counties.map(per_county)

    features = era5.map(per_image).flatten().getInfo()
    records = [f["properties"] for f in features["features"]]

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    return df

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main(mode: str):

    if mode == "historical":
        years = range(HIST_START_YEAR, HIST_END_YEAR + 1)
        out_dir = HISTORICAL_OUT_DIR

    elif mode == "live":
        years = [LIVE_YEAR]
        out_dir = LIVE_OUT_DIR

    else:
        raise ValueError("Mode must be 'historical' or 'live'")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []

    for year in years:
        print(f"\nExtracting {year}")
        for month in range(START_MONTH, END_MONTH + 1):
            df = extract_weather_for_year_month(year, month)
            all_frames.append(df)

    final_df = pd.concat(all_frames, ignore_index=True)

    final_path = out_dir / FINAL_FILENAME
    final_df.to_csv(final_path, index=False)

    print("\n✔ County-wise ERA5 extraction complete")
    print(f"Saved to: {final_path}")

# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["historical", "live"]
    )

    args = parser.parse_args()
    main(mode=args.mode)