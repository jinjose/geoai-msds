import boto3
from botocore.exceptions import ClientError
import json

import boto3, os
s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION","ap-south-1"))

DATA_BUCKET=os.environ["DATA_BUCKET"]
STATE_FIPS=os.environ["STATE_FIPS"]
COUNTY_FIPS=os.environ["COUNTY_FIPS"]

prefix = f"raw/dataset=yield/state_fips={STATE_FIPS}/county_fips={COUNTY_FIPS}/"
print("Searching prefix:", prefix)

keys=[]
token=None
while True:
    kwargs={"Bucket":DATA_BUCKET, "Prefix":prefix}
    if token: kwargs["ContinuationToken"]=token
    resp=s3.list_objects_v2(**kwargs)
    for obj in resp.get("Contents", []):
        keys.append(obj["Key"])
    if resp.get("IsTruncated"):
        token=resp["NextContinuationToken"]
    else:
        break

print("Found", len(keys), "objects")
print("\n".join(keys[:50]))

# def get_secret():
#     secret_name = "geoai/gee/service_account_json"
#     region_name = "ap-south-1"

#     client = boto3.client("secretsmanager", region_name=region_name)

#     try:
#         resp = client.get_secret_value(SecretId=secret_name)
#     except ClientError as e:
#         raise e

#     # SecretString should be the full JSON you pasted
#     secret_str = resp.get("SecretString")
#     if not secret_str:
#         raise ValueError("SecretString is empty. Did you store the JSON as plaintext?")

#     secret = json.loads(secret_str)
#     return secret  # ✅ this was missing

# if __name__ == "__main__":
#     print("Testing secret retrieval...")
#     secret_value = get_secret()
#     print("Secret value retrieved successfully.")
#     print("project_id:", secret_value.get("project_id"))
#     print("client_email:", secret_value.get("client_email"))