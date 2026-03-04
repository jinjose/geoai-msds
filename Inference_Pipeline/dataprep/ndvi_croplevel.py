import ee
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------
# 1. Initialize Earth Engine
# -----------------------------------------------------------
ee.Initialize(project="msds-432-gcp-cloud")

# -----------------------------------------------------------
# 2. Year range
# -----------------------------------------------------------
START_YEAR = 2024
END_YEAR = 2025  # CDL lags ~1 year

STATE_NAME = "illinois"
STATE_ID = "17"  # illinois = 17, iowa = 19

# -----------------------------------------------------------
# 3. Iowa counties
# -----------------------------------------------------------
counties = ee.FeatureCollection("TIGER/2018/Counties")
iowa_counties = counties.filter(ee.Filter.eq("STATEFP", STATE_ID))  # Iowa = 19

# -----------------------------------------------------------
# 4. Output directory
# -----------------------------------------------------------
OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# 5. Corn mask from USDA CDL
# -----------------------------------------------------------
def get_corn_mask(year: int):
    """
    CDL class code:
    1 = Corn
    """
    cdl = (
        ee.ImageCollection("USDA/NASS/CDL")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .first()
        .select("cropland")
    )
    return cdl.eq(1)

# -----------------------------------------------------------
# 6. Extract county-level CORN NDVI for one year
# -----------------------------------------------------------
def extract_corn_ndvi_for_year(year: int) -> pd.DataFrame:
    print(f"Extracting county-level CORN NDVI for {year}...")

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
    df["NDVI"] = df["NDVI"].astype(float) / 10000.0

    return df

# -----------------------------------------------------------
# 7. Extract multiple years
# -----------------------------------------------------------
def extract_multiple_years(start_year, end_year):
    all_frames = []

    for year in range(start_year, end_year + 1):
        df_year = extract_corn_ndvi_for_year(year)
        df_year.to_csv(
            OUT_DIR / f"corn_ndvi_{STATE_NAME}_county_{year}.csv",
            index=False
        )
        all_frames.append(df_year)

    final_df = pd.concat(all_frames, ignore_index=True)
    final_df.to_csv(
        OUT_DIR / f"corn_ndvi_{STATE_NAME}_county_all_years.csv",
        index=False
    )

    print("\n✔ County-level CORN NDVI extraction complete")
    print(f"Saved to: {OUT_DIR / f'corn_ndvi_{STATE_NAME}_county_all_years.csv'}")

# -----------------------------------------------------------
# 8. Run
# -----------------------------------------------------------
if __name__ == "__main__":
    extract_multiple_years(START_YEAR, END_YEAR)