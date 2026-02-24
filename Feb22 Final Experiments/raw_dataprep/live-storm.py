import requests
import pandas as pd
import logging
from datetime import datetime, timezone
from time import sleep
import time
import geopandas as gpd
from shapely.geometry import Point

# --------------------------------------------------
# Logging Setup
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Config
# --------------------------------------------------

NWS_BASE = "https://api.weather.gov"

HEADERS = {
    "User-Agent": "GeoAI-Yield-Project (your_real_email@example.com)"
}

REQUEST_TIMEOUT = 10

IOWA_COUNTY_SHP = "tl_2023_us_county.shp"  # Update path if needed


# --------------------------------------------------
# Load Iowa Counties
# --------------------------------------------------

def load_iowa_counties():
    logger.info("Loading Iowa county shapefile...")
    counties = gpd.read_file(IOWA_COUNTY_SHP)
    iowa = counties[counties["STATEFP"] == "19"].copy()
    iowa = iowa[["NAME", "geometry"]]
    iowa.rename(columns={"NAME": "county"}, inplace=True)
    return iowa


# --------------------------------------------------
# Get Iowa Stations
# --------------------------------------------------

def get_iowa_stations(limit=10):

    logger.info("Fetching Iowa stations...")

    url = f"{NWS_BASE}/stations?state=IA"

    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()

    stations = []

    for feature in data["features"]:
        stations.append({
            "station_id": feature["properties"]["stationIdentifier"],
            "latitude": feature["geometry"]["coordinates"][1],
            "longitude": feature["geometry"]["coordinates"][0]
        })

    stations_df = pd.DataFrame(stations).head(limit)

    logger.info(f"Using {len(stations_df)} stations for this run.")

    return stations_df


# --------------------------------------------------
# Get Today's Observations
# --------------------------------------------------

def get_today_observations(station_id):

    url = f"{NWS_BASE}/stations/{station_id}/observations"
    params = {"limit": 200}

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            logger.warning(f"Failed station {station_id}")
            return []

        data = response.json()
        today_utc = datetime.now(timezone.utc).date()
        results = []

        for feature in data["features"]:
            props = feature["properties"]
            timestamp = props.get("timestamp")

            if not timestamp:
                continue

            obs_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            if obs_time.date() != today_utc:
                continue

            wind_gust = props.get("windGust", {}).get("value")

            if wind_gust is None:
                continue

            wind_gust_mph = wind_gust * 2.23694

            results.append({
                "station_id": station_id,
                "timestamp": timestamp,
                "wind_gust_mph": wind_gust_mph
            })

        return results

    except Exception as e:
        logger.error(f"Error fetching {station_id}: {e}")
        return []


# --------------------------------------------------
# Main Fetch + County Mapping
# --------------------------------------------------

def fetch_iowa_today_gusts(threshold_mph=90):

    start_time = time.time()

    stations_df = get_iowa_stations(limit=10)
    all_results = []

    for idx, row in stations_df.iterrows():

        logger.info(f"Processing station {idx+1}/{len(stations_df)} | {row['station_id']}")

        station_results = get_today_observations(row["station_id"])

        for r in station_results:
            r["latitude"] = row["latitude"]
            r["longitude"] = row["longitude"]
            all_results.append(r)

        sleep(0.1)

    df = pd.DataFrame(all_results)

    if df.empty:
        logger.warning("No gust observations found.")
        return df, df, None

    # --------------------------------------------------
    # Map Stations to Counties
    # --------------------------------------------------

    logger.info("Mapping stations to counties...")

    counties_gdf = load_iowa_counties()

    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    stations_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    joined = gpd.sjoin(
        stations_gdf,
        counties_gdf,
        how="left",
        predicate="within"
    )

    # --------------------------------------------------
    # Aggregate County-Level Feature
    # --------------------------------------------------

    logger.info("Aggregating county-level storm exposure...")

    joined["severe_flag"] = (joined["wind_gust_mph"] >= threshold_mph).astype(int)

    county_agg = (
        joined
        .groupby("county")
        .agg(
            wind_tail_90_cutoff=("severe_flag", "sum"),
            max_gust_today=("wind_gust_mph", "max")
        )
        .reset_index()
    )

    elapsed = time.time() - start_time
    logger.info(f"Finished in {elapsed:.2f} seconds")

    logger.info(f"County-level storm feature sample:\n{county_agg.head()}")

    return joined, county_agg, df


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":

    logger.info("Starting Iowa storm ingestion...")

    station_level, county_level, raw_df = fetch_iowa_today_gusts(threshold_mph=90)

    logger.info("Done.")