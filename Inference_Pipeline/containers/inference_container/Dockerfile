FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py ./serve.py

# ✅ Create a "serve" command in PATH (because SageMaker is trying to exec "serve")
RUN printf '#!/bin/sh\nexec gunicorn -b 0.0.0.0:8080 serve:app --timeout 300\n' > /usr/local/bin/serve \
 && chmod +x /usr/local/bin/serve

EXPOSE 8080
ENV PYTHONUNBUFFERED=1

# SageMaker will still run "serve" (because your model config overrides CMD)
CMD ["serve"]