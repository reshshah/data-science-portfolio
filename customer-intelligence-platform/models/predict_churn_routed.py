#!/usr/bin/env python3
"""
Blends the logistic regression and LightGBM churn models by customer tenure.

Rationale: LightGBM wins on aggregate metrics but overfits patterns specific
to customers seen during training, so it generalizes poorly to customers with
little history. Logistic regression is weaker overall but far more robust on
that segment. Routing by tenure gets the best of both — see
docs/MODEL_METRICS_NOTES.md for the full comparison.

This script doesn't train anything; it loads the two already-trained
pipelines (run churn_config.yaml and churn_lightgbm_config.yaml through
models/train_classifier.py first) and evaluates the blended predictions.
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_split
from src.evaluator import best_threshold, compute_metrics
from src.reporting import print_summary, save_json, save_predictions
from src.routing import route_predict
from src.utils import get_project_root, load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the tenure-routed churn model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/churn_routed_config.yaml"),
        help="Path to the routed model config YAML (relative to project root).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = get_project_root()
    config = load_config(root / args.config)
    setup_logging(root / config["logging"]["file"], config["logging"]["level"])

    data_dir = root / config["paths"]["data_dir"]
    target = config["target"]
    tenure_threshold_days = config["tenure_threshold_days"]

    output_dir = root / config["paths"]["output_dir"] / config["name"] / "routed"
    metric_dir = output_dir / "metrics"
    pred_dir = output_dir / "predictions"
    for d in (metric_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    cold_start_pipe = joblib.load(root / config["paths"]["cold_start_model"])
    primary_pipe = joblib.load(root / config["paths"]["primary_model"])

    valid = load_split(data_dir, "validation")
    test = load_split(data_dir, "test")

    feature_cols = [c for c in valid.columns if c not in (target, "customer_id", "snapshot_date")]

    X_valid, y_valid = valid[feature_cols], valid[target].astype(int)
    X_test, y_test = test[feature_cols], test[target].astype(int)

    val_prob = route_predict(cold_start_pipe, primary_pipe, X_valid, valid["tenure_days"], tenure_threshold_days)
    threshold, _ = best_threshold(y_valid, val_prob, config)
    val_pred = (val_prob >= threshold).astype(int)
    val_metrics = compute_metrics(y_valid, val_pred, val_prob)
    val_metrics["threshold"] = threshold
    save_json(val_metrics, metric_dir / "validation_metrics.json")
    save_predictions(y_valid, val_prob, val_pred, pred_dir / "validation_predictions.csv")

    test_prob = route_predict(cold_start_pipe, primary_pipe, X_test, test["tenure_days"], tenure_threshold_days)
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)
    test_metrics["threshold"] = threshold
    save_json(test_metrics, metric_dir / "test_metrics.json")
    save_predictions(y_test, test_prob, test_pred, pred_dir / "test_predictions.csv")

    routed_rows = (test["tenure_days"] <= tenure_threshold_days).sum()
    logger.info("Routed %d / %d test rows to the cold-start model (tenure <= %d days)", routed_rows, len(test), tenure_threshold_days)

    print_summary(None, val_metrics, test_metrics, threshold, output_dir, title="Routed Evaluation Complete")


if __name__ == "__main__":
    main()
