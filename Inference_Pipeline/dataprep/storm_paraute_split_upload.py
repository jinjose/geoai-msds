import pandas as pd
import boto3
from io import BytesIO

INPUT_FILE = "storm_daily_all_year.parquet"
S3_BUCKET = "geoai-demo-data"
S3_PREFIX = "raw/dataset=storm/granularity=daily"

def split_and_upload():
    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    s3 = boto3.client("s3")

    for year, df_year in df.groupby("year"):
        buffer = BytesIO()
        df_year.to_parquet(buffer, index=False)
        buffer.seek(0)

        key = f"{S3_PREFIX}/year={year}/storm_daily_{year}.parquet"

        s3.upload_fileobj(buffer, S3_BUCKET, key)
        print(f"Uploaded {key}")

if __name__ == "__main__":
    split_and_upload()