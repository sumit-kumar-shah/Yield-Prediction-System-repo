"""
test_api.py
Integration tests for FastAPI endpoints using httpx test client.
Note: These tests require the model to be trained first.
      Run: python src/train.py  then  pytest tests/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

import pytest
from fastapi.testclient import TestClient

MODEL_PATH = Path(__file__).parent.parent / "models" / "ensemble_model.pkl"

# Skip API tests if model not trained
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Train model first: python src/train.py"
)


@pytest.fixture(scope="module")
def client():
    from app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_crops_endpoint(client):
    r = client.get("/crops")
    assert r.status_code == 200
    d = r.json()
    assert "crops" in d and "states" in d and "seasons" in d
    assert len(d["crops"]) > 0


def test_predict_valid(client):
    payload = {
        "state": "Punjab",
        "crop": "Wheat",
        "season": "Rabi",
        "year": 2023,
        "area_hectares": 1200.0,
        "rainfall_mm": 650.0,
        "fertilizer_kg_per_ha": 180.0,
        "avg_temp_c": 24.0,
        "irrigation_pct": 75.0,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    d = r.json()
    assert "predicted_yield_kg_per_ha" in d
    assert d["predicted_yield_kg_per_ha"] > 0
    assert d["confidence_low"] < d["predicted_yield_kg_per_ha"] < d["confidence_high"]


def test_predict_missing_field(client):
    payload = {"state": "Punjab", "crop": "Wheat"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_history_endpoint(client):
    r = client.get("/history?limit=5")
    assert r.status_code == 200
    assert "predictions" in r.json()


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    d = r.json()
    assert "test_r2" in d
    assert 0 < d["test_r2"] <= 1.0


def test_trend_endpoint(client):
    r = client.get("/trend?state=Punjab&crop=Wheat&season=Rabi")
    assert r.status_code == 200
    d = r.json()
    assert "years" in d and "yields" in d
    assert len(d["years"]) == len(d["yields"])
