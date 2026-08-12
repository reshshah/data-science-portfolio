#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_feature_metadata, load_split
from src.evaluator import best_threshold, compute_metrics
from src.feature_validation import validate_features
from src.plots import feature_importance, plot_feature_importance, plot_roc_pr
from src.preprocessing import split_xy
from src.reporting import print_summary, save_json, save_predictions
from src.trainer import train_dummy, train_model
from src.utils import get_project_root, load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary classifier from a config file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/churn_config.yaml"),
        help="Path to the model config YAML (relative to project root).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = get_project_root()
    config = load_config(root / args.config)
    setup_logging(root / config["logging"]["file"], config["logging"]["level"])

    data_dir = root / config["paths"]["data_dir"]
    metadata = load_feature_metadata(root / config["paths"]["metadata_file"])
    target = config["target"]
    model_type = config["model"]["type"]

    output_dir = root / config["paths"]["output_dir"] / config["name"] / model_type
    model_dir = output_dir / "models"
    metric_dir = output_dir / "metrics"
    pred_dir = output_dir / "predictions"
    plot_dir = output_dir / "plots"
    for d in (model_dir, metric_dir, pred_dir, plot_dir):
        d.mkdir(parents=True, exist_ok=True)

    train = load_split(data_dir, "train")
    valid = load_split(data_dir, "validation")
    test = load_split(data_dir, "test")

    for name, df in [("train", train), ("validation", valid), ("test", test)]:
        validate_features(df, metadata)

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    X_train, y_train = split_xy(train, target, numeric_features, categorical_features)
    X_valid, y_valid = split_xy(valid, target, numeric_features, categorical_features)
    X_test, y_test = split_xy(test, target, numeric_features, categorical_features)

    dummy = train_dummy(X_train, y_train)
    dummy_prob = dummy.predict_proba(X_valid)[:, 1]
    dummy_metrics = compute_metrics(y_valid, dummy.predict(X_valid), dummy_prob)
    save_json(dummy_metrics, metric_dir / "dummy_validation.json")

    pipe = train_model(X_train, y_train, numeric_features, categorical_features, config)

    val_prob = pipe.predict_proba(X_valid)[:, 1]
    threshold, _ = best_threshold(y_valid, val_prob, config)
    val_pred = (val_prob >= threshold).astype(int)
    val_metrics = compute_metrics(y_valid, val_pred, val_prob)
    val_metrics["threshold"] = threshold
    save_json(val_metrics, metric_dir / "validation_metrics.json")
    save_predictions(y_valid, val_prob, val_pred, pred_dir / "validation_predictions.csv")

    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)
    test_metrics["threshold"] = threshold
    save_json(test_metrics, metric_dir / "test_metrics.json")
    save_predictions(y_test, test_prob, test_pred, pred_dir / "test_predictions.csv")

    joblib.dump(pipe, model_dir / f"{model_type}.pkl")

    fi = feature_importance(pipe)
    fi.to_csv(metric_dir / "feature_importance.csv", index=False)
    plot_feature_importance(fi, plot_dir)
    plot_roc_pr(y_test, test_prob, plot_dir)

    print_summary(dummy_metrics, val_metrics, test_metrics, threshold, output_dir)


if __name__ == "__main__":
    main()
