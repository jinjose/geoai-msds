FROM python:3.10-slim
WORKDIR /opt/program
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY evaluate.py .
ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python","evaluate.py"]
