import json
import time
from typing import List, Optional, Tuple
import numpy as np
import onnxruntime as ort
from .config import AUDIT_PATH, DEMO_SAMPLE_PATH, EXPECTED_DAYS, METRICS_PATH, MODEL_PATH


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _priority_tier(prob: float) -> str:
    if prob >= 0.70:
        return "CRITICAL"
    if prob >= 0.45:
        return "HIGH"
    if prob >= 0.25:
        return "MEDIUM"
    return "LOW"


class GridGuardONNXEngine:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")
        if not METRICS_PATH.exists():
            raise FileNotFoundError(f"Metrics file not found at {METRICS_PATH}")

        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        self.audit = (
            json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
            if AUDIT_PATH.exists()
            else {}
        )
        self.mean = float(self.metrics["train_log1p_mean"])
        self.std = float(self.metrics["train_log1p_std"])

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.cached_demo_queue = self._build_demo_queue()

    def preprocess_batch(
        self, batch_series: List[List[Optional[float]]]
    ) -> Tuple[np.ndarray, List[dict]]:
        raw = np.array(
            [[np.nan if v is None else float(v) for v in seq] for seq in batch_series],
            dtype=np.float32,
        )
        if raw.ndim != 2 or raw.shape[1] != EXPECTED_DAYS:
            raise ValueError(f"Expected shape (N, {EXPECTED_DAYS}), got {raw.shape}")

        missing_mask = np.isnan(raw).astype(np.float32)
        non_neg = np.clip(np.nan_to_num(raw, nan=0.0), 0.0, None)

        scaled = (np.log1p(non_neg) - self.mean) / self.std
        tensor = np.stack([scaled, missing_mask], axis=1).reshape(-1, 2, 147, 7)

        signals = []
        for i in range(len(batch_series)):
            valid = non_neg[i][missing_mask[i] == 0]
            overall_mean = float(valid.mean()) if len(valid) > 0 else 0.0
            recent = non_neg[i, -90:][missing_mask[i, -90:] == 0]
            recent_mean = float(recent.mean()) if len(recent) > 0 else 0.0

            signals.append(
                {
                    "missing_day_ratio": round(float(missing_mask[i].mean()), 4),
                    "zero_day_ratio": round(float((non_neg[i] == 0).mean()), 4),
                    "mean_daily_kwh": round(overall_mean, 2),
                    "recent_90d_ratio": round(recent_mean / (overall_mean + 1e-6), 3),
                }
            )
        return tensor.astype(np.float32), signals

    def predict_batch(
        self, consumer_ids: List[str], batch_series: List[List[Optional[float]]]
    ) -> Tuple[List[dict], float]:
        t0 = time.perf_counter()
        tensor, signals = self.preprocess_batch(batch_series)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
        probs = _sigmoid(np.asarray(logits).reshape(-1))
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        results = []
        for cid, prob, sig in zip(consumer_ids, probs, signals):
            p = float(prob)
            results.append(
                {
                    "consumer_id": cid,
                    "theft_probability": round(p, 4),
                    "risk_score_100": round(p * 100.0, 1),
                    "priority_tier": _priority_tier(p),
                    "signals": sig,
                    "inference_ms": elapsed_ms,
                }
            )
        results.sort(key=lambda r: r["theft_probability"], reverse=True)
        return results, elapsed_ms

    def _build_demo_queue(self) -> List[dict]:
        if not DEMO_SAMPLE_PATH.exists():
            return []
        records = json.loads(DEMO_SAMPLE_PATH.read_text(encoding="utf-8"))
        cids = [r["consumer_id"] for r in records]
        series = [r["daily_kwh"] for r in records]
        scored, _ = self.predict_batch(cids, series)

        gt_map = {r["consumer_id"]: r.get("ground_truth_flag") for r in records}
        for rank, item in enumerate(scored, start=1):
            item["inspection_rank"] = rank
            item["ground_truth_flag"] = gt_map.get(item["consumer_id"])
        return scored
