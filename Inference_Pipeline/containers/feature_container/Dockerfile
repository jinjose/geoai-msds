FROM python:3.10-slim
WORKDIR /opt/program
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run_feature_builder.py ./
COPY utils.py ./
COPY app ./app

# ✅ Verify schema files are inside the image (build will show paths)
RUN echo "=== feature_schema.json files in image ===" && \
    find /opt/program/app -maxdepth 2 -name "feature_schema.json" -print && \
    echo "=== list season folders ===" && \
    ls -la /opt/program/app/jun01 /opt/program/app/jul01 /opt/program/app/aug01 || true

ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python","run_feature_builder.py"]
