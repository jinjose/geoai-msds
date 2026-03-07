FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY docker/requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

# App code: copy src package and batch runner
COPY src/geoai /opt/program/geoai
COPY docker/batch_runner.py /opt/program/batch_runner.py

# Add 'serve' so SageMaker can exec it
# COPY docker/serve /usr/local/bin/serve
# RUN chmod +x /usr/local/bin/serve
#ENTRYPOINT ["serve"]
# Make 'serve' the default entrypoint (SageMaker expects it)
#ENTRYPOINT ["python", "/opt/program/batch_runner.py"]
# Keep your entrypoint/CMD as you designed
#CMD ["python", "batch_runner.py"]
# Copy serve.py into the image
COPY docker/serve.py /opt/program/serve.py
# EXPOSE 8080
# # Start the SageMaker inference server
# CMD ["python", "serve.py"]
# Add SageMaker "serve" entrypoint expected by Batch Transform / Hosting
COPY docker/serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve && sed -i 's/\r$//' /usr/local/bin/serve

EXPOSE 8080
ENV PORT=8080

# SageMaker will run: serve
CMD ["serve"]



