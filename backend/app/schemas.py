import math
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from .config import EXPECTED_DAYS


class PredictRequest(BaseModel):
    consumer_id: str = Field(..., min_length=1, max_length=128)
    daily_kwh: List[Optional[float]] = Field(
        ...,
        min_length=EXPECTED_DAYS,
        max_length=EXPECTED_DAYS,
        description="1,029 chronological daily kWh readings (null for missing days).",
    )

    @field_validator("daily_kwh")
    @classmethod
    def validate_finite_values(cls, values: List[Optional[float]]) -> List[Optional[float]]:
        for v in values:
            if v is not None and not math.isfinite(v):
                raise ValueError("daily_kwh must contain finite numbers or null.")
        return values


class BatchPredictRequest(BaseModel):
    consumers: List[PredictRequest] = Field(..., min_length=1, max_length=500)


class SignalSummary(BaseModel):
    missing_day_ratio: float
    zero_day_ratio: float
    mean_daily_kwh: float
    recent_90d_ratio: float


class PredictResponse(BaseModel):
    consumer_id: str
    theft_probability: float
    risk_score_100: float
    priority_tier: str
    signals: SignalSummary
    inference_ms: float
    disclaimer: str


class BatchPredictResponse(BaseModel):
    count: int
    total_inference_ms: float
    ranked_results: List[PredictResponse]
    disclaimer: str
