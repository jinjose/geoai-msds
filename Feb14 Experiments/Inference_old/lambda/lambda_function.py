import boto3
import os
from datetime import datetime

sm = boto3.client("sagemaker")

def lambda_handler(event, context):
    job_name = "geoai-batch-" + datetime.utcnow().strftime("%Y%m%d%H%M")

    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=os.environ["MODEL_NAME"],
        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3Uri": os.environ["INPUT_S3"],
                    "S3DataType": "S3Prefix"
                }
            },
            "ContentType": "text/csv"
        },
        TransformOutput={
            "S3OutputPath": os.environ["OUTPUT_S3"]
        },
        TransformResources={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1
        }
    )

    return {"status": "Transform started"}
