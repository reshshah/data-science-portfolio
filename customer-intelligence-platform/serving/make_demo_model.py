"""Train a demo churn model and register it, so the serving layer runs end-to-end.

This stands in for the real training pipeline (models/train_classifier.py).
It generates synthetic customer features, trains a logistic regression,
evaluates it on a holdout set, and writes a versioned bundle to the registry.

Run from the customer-intelligence-platform directory:

    python -m serving.make_demo_model
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "recency_days",
    "frequency_90d",
    "monetary_90d",
    "tenure_days",
    "support_tickets_90d",
    "web_sessions_30d",
]
TARGET = "churn_label_180d"


def make_synthetic_customers(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Synthetic customers where churn rises with recency and support tickets."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "recency_days": rng.exponential(30, n),
        "frequency_90d": rng.poisson(4, n),
        "monetary_90d": rng.gamma(2, 60, n),
        "tenure_days": rng.uniform(30, 2000, n),
        "support_tickets_90d": rng.poisson(0.5, n),
        "web_sessions_30d": rng.poisson(6, n),
    })
    logit = (
        0.03 * df["recency_days"]
        - 0.25 * df["frequency_90d"]
        - 0.004 * df["monetary_90d"]
        - 0.0008 * df["tenure_days"]
        + 0.6 * df["support_tickets_90d"]
        - 0.08 * df["web_sessions_30d"]
        - 0.2
    )
    p = 1 / (1 + np.exp(-logit))
    df[TARGET] = rng.binomial(1, p)
    return df


def train_and_register(registry_dir, n: int = 2000, seed: int = 7) -> dict:
    """Train, evaluate, and write a new immutable version to the registry."""
    df = make_synthetic_customers(n=n, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.25, random_state=seed, stratify=df[TARGET]
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.split("_v")[1]) for p in registry_dir.glob("model_v*") if p.is_dir()]
    version = f"model_v{max(existing, default=0) + 1}"
    vdir = registry_dir / version
    vdir.mkdir()

    with open(vdir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    (vdir / "feature_metadata.json").write_text(json.dumps({
        "feature_names": FEATURES,
        "target": TARGET,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(X_train),
        "data_source": "synthetic (make_demo_model.py)",
    }, indent=2))
    (vdir / "metrics.json").write_text(json.dumps({
        "auc_test": round(float(auc), 4),
        "n_test": len(X_test),
        "base_churn_rate": round(float(df[TARGET].mean()), 4),
    }, indent=2))

    print(f"Registered {version} — holdout AUC {auc:.3f} — at {vdir}")
    return {"version": version, "auc_test": auc, "path": str(vdir)}


if __name__ == "__main__":
    default_registry = Path(__file__).resolve().parents[1] / "registry"
    train_and_register(default_registry)
