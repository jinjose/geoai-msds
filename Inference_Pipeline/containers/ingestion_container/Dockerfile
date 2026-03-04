FROM python:3.10-slim

WORKDIR /opt/program

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY geoai_ingest/ ./geoai_ingest/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python","-m","geoai_ingest.cli"]
