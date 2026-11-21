import logging
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from .config import ALLOWED_ORIGINS, API_KEY, LOG_LEVEL, SCREENING_DISCLAIMER
from .inference import GridGuardONNXEngine
from .schemas import BatchPredictRequest, BatchPredictResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("gridguard.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading GridGuard ONNX inference engine...")
    app.state.engine = GridGuardONNXEngine()
    logger.info(
        "Engine ready | model=%s | cached_demo=%d",
        app.state.engine.metrics.get("model"),
        len(app.state.engine.cached_demo_queue),
    )
    yield
    logger.info("Shutting down GridGuard API.")


app = FastAPI(
    title="GridGuard AI Production API",
    version="2.0.0",
    description="2-Channel 2D-CNN ONNX electricity theft screening & inspection-triage API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000.0, 2)
    logger.info("%s %s -> %d (%.2f ms)", request.method, request.url.path, response.status_code, ms)
    return response


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


def get_engine(request: Request) -> GridGuardONNXEngine:
    return request.app.state.engine


@app.get("/health")
def health(engine: GridGuardONNXEngine = Depends(get_engine)):
    return {
        "status": "ok",
        "model": engine.metrics.get("model"),
        "roc_auc": engine.metrics.get("test_roc_auc"),
        "pr_auc": engine.metrics.get("test_pr_auc"),
    }


@app.get("/api/model-metrics")
def model_metrics(engine: GridGuardONNXEngine = Depends(get_engine)):
    return {
        "metrics": engine.metrics,
        "data_audit": engine.audit,
        "disclaimer": SCREENING_DISCLAIMER,
    }


@app.get("/api/demo/houses")
def demo_houses(top_k: int = 50, engine: GridGuardONNXEngine = Depends(get_engine)):
    k = max(1, min(top_k, len(engine.cached_demo_queue)))
    return {
        "total_cached": len(engine.cached_demo_queue),
        "returned": k,
        "houses": engine.cached_demo_queue[:k],
        "disclaimer": SCREENING_DISCLAIMER,
    }


@app.post("/api/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
def predict_single(req: PredictRequest, engine: GridGuardONNXEngine = Depends(get_engine)):
    results, _ = engine.predict_batch([req.consumer_id], [req.daily_kwh])
    return PredictResponse(**results[0], disclaimer=SCREENING_DISCLAIMER)


@app.post("/api/predict/batch", response_model=BatchPredictResponse, dependencies=[Depends(verify_api_key)])
def predict_batch(req: BatchPredictRequest, engine: GridGuardONNXEngine = Depends(get_engine)):
    cids = [c.consumer_id for c in req.consumers]
    series = [c.daily_kwh for c in req.consumers]
    results, total_ms = engine.predict_batch(cids, series)
    return BatchPredictResponse(
        count=len(results),
        total_inference_ms=total_ms,
        ranked_results=[PredictResponse(**r, disclaimer=SCREENING_DISCLAIMER) for r in results],
        disclaimer=SCREENING_DISCLAIMER,
    )
