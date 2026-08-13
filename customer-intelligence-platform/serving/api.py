"""Real-time scoring API.

Loads the latest model from the registry at startup and serves predictions.

Run from the customer-intelligence-platform directory:

    uvicorn serving.api:app --reload

Endpoints:
    GET  /health   → model version + status (for load balancers / k8s probes)
    POST /predict  → churn score for one customer

Set REGISTRY_DIR to point at a non-default registry (used by tests and Docker).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model_loader import load_model, validate_features

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "registry"

state = {"bundle": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = Path(os.environ.get("REGISTRY_DIR", DEFAULT_REGISTRY))
    state["bundle"] = load_model(registry)
    yield
    state["bundle"] = None


app = FastAPI(title="Customer Intelligence Scoring API", lifespan=lifespan)


class PredictRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    features: dict[str, float]


class PredictResponse(BaseModel):
    customer_id: str
    churn_score: float
    model_version: str


@app.get("/health")
def health():
    bundle = state["bundle"]
    return {
        "status": "ok",
        "model_version": bundle.version,
        "auc_at_training": bundle.metrics.get("auc_test"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    bundle = state["bundle"]
    try:
        row = validate_features(bundle, req.features)
    except ValueError as exc:
        # Reject contract violations loudly — never impute silently in serving.
        raise HTTPException(status_code=400, detail=str(exc))
    # DataFrame (not a bare list) so feature names match what the model saw in training
    X = pd.DataFrame([row], columns=bundle.feature_names)
    score = float(bundle.model.predict_proba(X)[0, 1])
    return PredictResponse(
        customer_id=req.customer_id,
        churn_score=round(score, 4),
        model_version=bundle.version,
    )
