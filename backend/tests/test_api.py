import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["roc_auc"] >= 0.80


def test_model_metrics_endpoint(client):
    res = client.get("/api/model-metrics")
    assert res.status_code == 200
    body = res.json()
    assert "metrics" in body
    assert body["metrics"]["test_pr_auc"] > body["metrics"]["baseline_pr_auc"] * 4


def test_single_predict_preserves_missing_mask(client):
    payload = {
        "consumer_id": "SGCC_TEST_001",
        "daily_kwh": [None if i % 10 == 0 else 14.2 for i in range(1029)],
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert 0.0 <= body["theft_probability"] <= 1.0
    assert body["signals"]["missing_day_ratio"] > 0.09
    assert body["signals"]["zero_day_ratio"] == 0.0
    assert "not proof of electricity theft" in body["disclaimer"].lower()


def test_batch_predict_returns_ranked_order(client):
    payload = {
        "consumers": [
            {"consumer_id": "NORMAL_LIKE", "daily_kwh": [15.0] * 1029},
            {"consumer_id": "DROP_LIKE", "daily_kwh": [20.0] * 700 + [0.0] * 329},
        ]
    }
    res = client.post("/api/predict/batch", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 2
    probs = [r["theft_probability"] for r in body["ranked_results"]]
    assert probs == sorted(probs, reverse=True)


def test_invalid_length_rejected_with_422(client):
    res = client.post("/api/predict", json={"consumer_id": "BAD", "daily_kwh": [1.0, 2.0]})
    assert res.status_code == 422


def test_demo_houses_queue_ranked(client):
    res = client.get("/api/demo/houses?top_k=10")
    assert res.status_code == 200
    body = res.json()
    assert body["returned"] == 10
    probs = [h["theft_probability"] for h in body["houses"]]
    assert probs == sorted(probs, reverse=True)
