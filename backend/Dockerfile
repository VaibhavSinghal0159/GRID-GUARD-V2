FROM python:3.11-slim

WORKDIR /app
RUN useradd --create-home appuser

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY ml/artifacts /app/ml/artifacts
COPY backend/app /app/backend/app

ENV PYTHONPATH=/app/backend \
    ARTIFACTS_DIR=/app/ml/artifacts \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
