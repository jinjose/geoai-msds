FROM python:3.10-slim

WORKDIR /opt/program

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/geoai/inference /opt/program
COPY model_artifacts /opt/ml/model

ENV SAGEMAKER_PROGRAM inference.py

ENTRYPOINT ["python", "inference.py"]
