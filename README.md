# GridGuard AI v2 -- Production Electricity Theft Detection System

End-to-end electricity theft screening and field-inspection triage microservice powered by a **2-Channel 2D-CNN** trained on the full multi-year SGCC smart-meter dataset (42,372 consumers x 1,029 days), exported to **ONNX**, and served via **FastAPI + ONNX Runtime** inside a hardened non-root Docker container.

## Architecture
SGCC Multi-Year CSV (1,029 days) -> PyTorch 2-Channel 2D-CNN (147x7 grid + missing mask) -> ONNX Export (opset 17) -> FastAPI + ONNX Runtime -> Docker / GitHub Actions CI

## Holdout Test Benchmarks (Untouched 20% Consumer Split)

| Metric | v1 (26-Day GBDT Baseline) | v2 (1,029-Day 2-Channel 2D-CNN) |
| :--- | :---: | :---: |
| **ROC-AUC** | 0.6455 | **0.8340** |
| **PR-AUC** (baseline 0.0853) | 0.1768 | **0.4233** (~5x lift) |
| **Top-10 Precision** | 80.0% | **90.0%** |
| **Top-50 Precision** | 40.0% | **82.0%** |
| **Top-100 Precision** | 34.0% | **75.0%** |
| **Top-500 Precision** | 19.0% | **51.2%** |

## Key Engineering Highlights
- **Missingness Preservation:** Reshapes 1,029 chronological daily readings into (2, 147, 7) (weeks x days). Channel 0 holds standardized log1p(kWh) and Channel 1 holds a binary missing-value mask so meter outages and tampering gaps are never erased.
- **Zero Data Leakage:** Strict consumer-level 60/20/20 split; standardization mean and std are fit exclusively on training consumers and verified for numerical parity between PyTorch and ONNX Runtime.
- **Lightweight Serving:** Zero PyTorch dependency in production; CPU inference via onnxruntime, strict Pydantic input validation, request latency middleware, and optional X-API-Key protection.
- **DevOps & Safety:** Non-root Docker image with /health healthcheck, automated GitHub Actions Pytest CI, and explicit human-in-the-loop screening disclaimers.

## Quick Start

Option A (Docker):

    docker compose up --build

Option B (Local Python):

    cd backend
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    $env:PYTHONPATH="."
    pytest tests/test_api.py -v
    uvicorn app.main:app --host 0.0.0.0 --port 8000

## API Endpoints
- GET /health -- Liveness check & loaded model metadata
- GET /api/model-metrics -- Holdout metrics & data audit summary
- GET /api/demo/houses?top_k=10 -- Top-K ranked inspection queue (50 real SGCC profiles)
- POST /api/predict -- Score a single 1,029-day consumer profile
- POST /api/predict/batch -- Vectorized batch scoring & priority ranking (up to 500 consumers)

## Operational & Ethical Scope
GridGuard AI is a **screening priority tool only** -- anomaly probabilities are not legal proof of theft and require physical field inspection before any billing or legal action. Trained on SGCC telemetry; deployment to Indian DISCOMs requires local domain fine-tuning and temporal drift monitoring.
