FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install deps from processing folder
COPY docker/processing/requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

# Copy processing scripts
COPY docker/processing/evaluation_processor.py /opt/program/evaluation_processor.py
COPY docker/processing/compare_and_report.py /opt/program/compare_and_report.py
COPY docker/processing/entrypoint.sh /opt/program/entrypoint.sh

RUN chmod +x /opt/program/entrypoint.sh

ENTRYPOINT ["/opt/program/entrypoint.sh"]