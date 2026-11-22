FROM python:3.11-slim

WORKDIR /app
RUN useradd --create-home appuser

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY ml/artifacts /app/ml/artifacts
COPY backend/app /app/backend/app

ENV PYTHONPATH=/app/backend \
    ARTIFACTS_DIR=/app/ml/artifacts \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os, urllib.request; p = os.environ.get('PORT', '8000'); urllib.request.urlopen('http://127.0.0.1:' + p + '/health')" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]