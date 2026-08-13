"""Batch scoring job: score every customer in a CSV, write a scores table.

This is the deployment mode that fits churn best — scores change slowly and
are consumed by campaign tools, so a scheduled job writing a table beats a
real-time endpoint.

Run from the customer-intelligence-platform directory:

    python -m serving.batch_score --input customers.csv --output scores.csv
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .model_loader import load_model

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "registry"


def score_file(input_csv, output_csv, registry_dir=DEFAULT_REGISTRY, version=None,
               id_column: str = "customer_id") -> pd.DataFrame:
    """Score all rows in input_csv; write id, score, model version, timestamp."""
    bundle = load_model(registry_dir, version)
    df = pd.read_csv(input_csv)

    missing = [c for c in [id_column] + bundle.feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    X = df[bundle.feature_names]  # training order, enforced by the bundle
    scores = bundle.model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        id_column: df[id_column],
        "churn_score": scores.round(4),
        "model_version": bundle.version,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    })
    out.to_csv(output_csv, index=False)
    print(f"Scored {len(out)} customers with {bundle.version} → {output_csv}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Batch-score customers for churn")
    parser.add_argument("--input", required=True, help="CSV with customer_id + feature columns")
    parser.add_argument("--output", required=True, help="Where to write the scores CSV")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--version", default=None, help="e.g. model_v1 (default: latest)")
    args = parser.parse_args()
    score_file(args.input, args.output, args.registry, args.version)


if __name__ == "__main__":
    main()
