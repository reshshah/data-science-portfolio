"""Tests for the serving layer: registry, feature contract, API, batch job.

Run from the customer-intelligence-platform directory: pytest tests/ -v
"""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from serving.batch_score import score_file
from serving.make_demo_model import FEATURES, make_synthetic_customers, train_and_register
from serving.model_loader import load_model, validate_features

VALID_FEATURES = {
    "recency_days": 45.0,
    "frequency_90d": 2.0,
    "monetary_90d": 120.0,
    "tenure_days": 400.0,
    "support_tickets_90d": 1.0,
    "web_sessions_30d": 3.0,
}


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """A registry with one trained demo model."""
    registry_dir = tmp_path_factory.mktemp("registry")
    result = train_and_register(registry_dir, n=1000, seed=7)
    assert result["auc_test"] > 0.65, "demo model should beat random comfortably"
    return registry_dir


@pytest.fixture(scope="module")
def client(registry, tmp_path_factory):
    from fastapi.testclient import TestClient

    import os
    os.environ["REGISTRY_DIR"] = str(registry)
    from serving.api import app
    with TestClient(app) as c:
        yield c


# --- registry & contract ---

def test_load_model_latest(registry):
    bundle = load_model(registry)
    assert bundle.version == "model_v1"
    assert bundle.feature_names == FEATURES
    assert "auc_test" in bundle.metrics


def test_new_training_run_creates_new_version(registry):
    result = train_and_register(registry, n=500, seed=8)
    assert result["version"] == "model_v2"
    assert load_model(registry).version == "model_v2"  # latest wins
    assert load_model(registry, "model_v1").version == "model_v1"  # old still loadable


def test_validate_features_rejects_missing_and_unexpected(registry):
    bundle = load_model(registry)
    with pytest.raises(ValueError, match="missing"):
        validate_features(bundle, {"recency_days": 1.0})
    with pytest.raises(ValueError, match="unexpected"):
        validate_features(bundle, {**VALID_FEATURES, "made_up_feature": 1.0})


def test_validate_features_orders_correctly(registry):
    bundle = load_model(registry)
    shuffled = dict(reversed(list(VALID_FEATURES.items())))
    row = validate_features(bundle, shuffled)
    assert row == [VALID_FEATURES[name] for name in bundle.feature_names]


# --- API ---

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_version"].startswith("model_v")


def test_predict_returns_probability(client):
    resp = client.post("/predict", json={"customer_id": "c123", "features": VALID_FEATURES})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_score"] <= 1.0
    assert body["customer_id"] == "c123"


def test_predict_rejects_bad_contract(client):
    resp = client.post("/predict", json={"customer_id": "c123", "features": {"recency_days": 1.0}})
    assert resp.status_code == 400
    assert "missing" in resp.json()["detail"]


def test_predict_rejects_malformed_request(client):
    resp = client.post("/predict", json={"features": VALID_FEATURES})  # no customer_id
    assert resp.status_code == 422


def test_risky_customer_scores_higher(client):
    low_risk = {**VALID_FEATURES, "recency_days": 2.0, "frequency_90d": 12.0,
                "support_tickets_90d": 0.0}
    high_risk = {**VALID_FEATURES, "recency_days": 150.0, "frequency_90d": 0.0,
                 "support_tickets_90d": 5.0}
    s_low = client.post("/predict", json={"customer_id": "a", "features": low_risk}).json()
    s_high = client.post("/predict", json={"customer_id": "b", "features": high_risk}).json()
    assert s_high["churn_score"] > s_low["churn_score"]


# --- batch job ---

def test_batch_score_file(registry, tmp_path):
    df = make_synthetic_customers(n=50, seed=11)
    df.insert(0, "customer_id", [f"c{i}" for i in range(len(df))])
    input_csv = tmp_path / "customers.csv"
    output_csv = tmp_path / "scores.csv"
    df.to_csv(input_csv, index=False)

    out = score_file(input_csv, output_csv, registry_dir=registry)
    assert output_csv.exists()
    assert len(out) == 50
    assert out["churn_score"].between(0, 1).all()
    assert set(out.columns) == {"customer_id", "churn_score", "model_version", "scored_at"}


def test_batch_score_rejects_missing_columns(registry, tmp_path):
    input_csv = tmp_path / "bad.csv"
    pd.DataFrame({"customer_id": ["c1"], "recency_days": [10.0]}).to_csv(input_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        score_file(input_csv, tmp_path / "out.csv", registry_dir=registry)
