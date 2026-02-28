FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NIDS_MODEL_PATH=artifacts_baseline/nids_model.joblib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_nids.py .
COPY artifacts_baseline/nids_model.joblib artifacts_baseline/nids_model.joblib

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn api_nids:app --host 0.0.0.0 --port ${PORT:-8000}"]
