FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    boto3 pandas pyarrow earthengine-api google-auth requests

COPY docker/ingest/ingest_raw.py /app/ingest_raw.py

ENTRYPOINT ["python", "/app/ingest_raw.py"]