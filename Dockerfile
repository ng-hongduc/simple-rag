FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY sql ./sql
COPY main.py .

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
