import boto3, json

def secrets_json(secret_id: str, region: str) -> dict:
    sm = boto3.client("secretsmanager", region_name=region)
    return json.loads(sm.get_secret_value(SecretId=secret_id)["SecretString"])

def secrets_text(secret_id: str, region: str) -> str:
    sm = boto3.client("secretsmanager", region_name=region)
    return sm.get_secret_value(SecretId=secret_id)["SecretString"].strip()
