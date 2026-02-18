import ee
import pandas as pd
from pathlib import Path

ee.Initialize(project="XXXXXXXX")

START_YEAR = 2013
END_YEAR = 2024

# Iowa counties
counties = ee.FeatureCollection("TIGER/2018/Counties")
iowa_counties = counties.filter(ee.Filter.eq("STATEFP", "19"))

OUT_DIR = Path("<YOUR_DIR>")
OUT_DIR.mkdir(parents=True, exist_ok=True)

weather_vars = {
    "temperature": "temperature_2m",
    "rainfall": "total_precipitation_sum",
    "evapotranspiration": "total_evaporation_sum",
    "soil_moisture": "volumetric_soil_water_layer_1"
}

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

def extract_multiple_years(start_year, end_year):
    all_frames = []

    for year in range(start_year, end_year + 1):
        print(f"\nExtracting {year}")
        for month in range(1, 13):
            df = extract_weather_for_year_month(year, month)
            all_frames.append(df)

    final_df = pd.concat(all_frames, ignore_index=True)
    final_df.to_csv(OUT_DIR / "era5_iowa_county_all_years.csv", index=False)

    print("\n✔ County-wise ERA5 extraction complete")

if __name__ == "__main__":
    extract_multiple_years(START_YEAR, END_YEAR)
