import ee
import pandas as pd
from pathlib import Path
import argparse

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
PROJECT_ID = "XXXXXXXX"

HIST_START_YEAR = 2013
HIST_END_YEAR = 2024
LIVE_YEAR = 2025
LATEST_CDL_YEAR = 2024

# Same filename for both modes
FINAL_FILENAME = "corn_ndvi_iowa_state.csv"

# Hard-coded directories
HISTORICAL_OUT_DIR = Path("training-dataset/raw")
LIVE_OUT_DIR = Path("inference-dataset/raw")

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
# Corn mask with fallback
# -----------------------------------------------------------
def get_corn_mask(year: int):
    mask_year = min(year, LATEST_CDL_YEAR)

    cdl = (
        ee.ImageCollection("USDA/NASS/CDL")
        .filterDate(f"{mask_year}-01-01", f"{mask_year}-12-31")
        .first()
        .select("cropland")
    )

    return cdl.eq(1)

# -----------------------------------------------------------
# Extract NDVI for one year
# -----------------------------------------------------------
def extract_corn_ndvi_for_year(year: int) -> pd.DataFrame:

    print(f"Extracting CORN NDVI for {year} using CDL {min(year, LATEST_CDL_YEAR)} mask...")

    corn_mask = get_corn_mask(year)

    modis = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select("NDVI")
    )

    def extract_per_image(image):

        masked = image.updateMask(corn_mask)

        def per_county(county):
            stats = masked.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=county.geometry(),
                scale=250,
                bestEffort=True,
                maxPixels=1e13
            )

            return ee.Feature(
                None,
                {
                    "date": image.date().format("YYYY-MM-dd"),
                    "year": year,
                    "county_name": county.get("NAME"),
                    "geoid": county.get("GEOID"),
                    "NDVI": stats.get("NDVI")
                }
            )

        return iowa_counties.map(per_county)

    features = modis.map(extract_per_image).flatten().getInfo()
    records = [f["properties"] for f in features["features"]]

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce") / 10000.0

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
        raise ValueError("Mode must be either 'historical' or 'live'")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []

    for year in years:
        df_year = extract_corn_ndvi_for_year(year)
        all_frames.append(df_year)

    final_df = pd.concat(all_frames, ignore_index=True)

    final_path = out_dir / FINAL_FILENAME
    final_df.to_csv(final_path, index=False)

    print("\n✔ Extraction complete")
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