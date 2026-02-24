import requests
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

BASE_URL = (
    "https://services.arcgis.com/jIL9msH9OI208GCb/arcgis/rest/services/"
    "NOAA_Storm_Events_Database_1950-2021_v2/FeatureServer/0/query"
)


# ============================================================
# STEP 1: GET OBJECT IDS (POST SAFE)
# ============================================================

def fetch_object_ids():

    where_clause = (
        "STATE = 'IOWA' "
        "AND YEAR >= 2013 "
        "AND YEAR <= 2024 "
        "AND (EVENT_TYPE = 'Thunderstorm Wind' OR EVENT_TYPE = 'High Wind')"
    )

    params = {
        "where": where_clause,
        "returnIdsOnly": "true",
        "f": "json"
    }

    response = requests.post(BASE_URL, data=params)
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(f"ArcGIS error: {data}")

    return data.get("objectIds", [])


# ============================================================
# STEP 2: FETCH RECORDS IN SAFE CHUNKS (POST + SMALLER CHUNK)
# ============================================================

def fetch_records_by_ids(object_ids, chunk_size=100):

    all_rows = []

    for i in range(0, len(object_ids), chunk_size):

        chunk = object_ids[i:i + chunk_size]
        id_string = ",".join(map(str, chunk))

        where_clause = f"OBJECTID IN ({id_string})"

        params = {
            "where": where_clause,
            "outFields": "*",
            "returnGeometry": "false",
            "f": "json"
        }

        response = requests.post(BASE_URL, data=params)
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            raise RuntimeError(f"ArcGIS error during chunk fetch: {data}")

        features = data.get("features", [])

        for feat in features:
            all_rows.append(feat["attributes"])

        print(f"Fetched {len(all_rows)} rows...")

    return pd.DataFrame(all_rows)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Getting Iowa wind event IDs (2013–2024)...")

    ids = fetch_object_ids()

    print(f"Total IDs found: {len(ids)}")

    if not ids:
        print("No records found.")
        exit()

    print("Downloading records in chunks...")
    df = fetch_records_by_ids(ids)

    print(f"Total rows downloaded: {len(df)}")

    # Convert date column
    if "BEGIN_DATE_TIME" in df.columns:
        df["BEGIN_DATE_TIME"] = pd.to_datetime(
            df["BEGIN_DATE_TIME"],
            errors="coerce"
        )

    # Normalize county
    if "CZ_NAME" in df.columns:
        df["county"] = (
            df["CZ_NAME"]
            .str.lower()
            .str.replace(" county", "", regex=False)
            .str.strip()
        )

    # Save file
    raw_dir = Path(__file__).resolve().parents[1] / "training-dataset" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / "iowa_storm_events.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")