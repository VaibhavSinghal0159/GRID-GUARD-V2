import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(BASE_DIR / "ml" / "artifacts")))

MODEL_PATH = ARTIFACTS_DIR / "gridguard_cnn.onnx"
METRICS_PATH = ARTIFACTS_DIR / "cnn_metrics.json"
AUDIT_PATH = ARTIFACTS_DIR / "data_audit.json"
DEMO_SAMPLE_PATH = ARTIFACTS_DIR / "demo_sample_50.json"

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://localhost:5173").split(",")
    if o.strip()
]
API_KEY = os.getenv("API_KEY", "").strip()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

EXPECTED_DAYS = 1029
SCREENING_DISCLAIMER = (
    "Screening priority score only — not proof of electricity theft. "
    "Physical field inspection and meter verification are required before any billing or legal action."
)
