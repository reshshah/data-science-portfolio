#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_feature_metadata, load_split
from src.evaluator import compute_regression_metrics
from src.feature_validation import validate_features
from src.plots import feature_importance, plot_feature_importance, plot_predicted_vs_actual
from src.preprocessing import split_xy
from src.reporting import save_json
from src.trainer import train_dummy_regressor, train_regression_model
from src.utils import get_project_root, load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a regression model from a config file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/clv_config.yaml"),
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
    plot_dir = output_dir / "plots"
    for d in (model_dir, metric_dir, plot_dir):
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

    dummy = train_dummy_regressor(X_train, y_train)
    dummy_pred = dummy.predict(X_valid)
    dummy_metrics = compute_regression_metrics(y_valid, dummy_pred)
    save_json(dummy_metrics, metric_dir / "dummy_validation.json")

    pipe = train_regression_model(X_train, y_train, numeric_features, categorical_features, config)

    val_pred = pipe.predict(X_valid)
    val_metrics = compute_regression_metrics(y_valid, val_pred)
    save_json(val_metrics, metric_dir / "validation_metrics.json")

    test_pred = pipe.predict(X_test)
    test_metrics = compute_regression_metrics(y_test, test_pred)
    save_json(test_metrics, metric_dir / "test_metrics.json")

    joblib.dump(pipe, model_dir / f"{model_type}.pkl")

    fi = feature_importance(pipe)
    fi.to_csv(metric_dir / "feature_importance.csv", index=False)
    plot_feature_importance(fi, plot_dir)
    plot_predicted_vs_actual(y_test, test_pred, plot_dir)

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info("Dummy Validation RMSE : %.2f", dummy_metrics["rmse"])
    logger.info("Validation RMSE        : %.2f", val_metrics["rmse"])
    logger.info("Validation MAE         : %.2f", val_metrics["mae"])
    logger.info("Validation R2          : %.3f", val_metrics["r2"])
    logger.info("Validation WAPE        : %.3f", val_metrics["wape"])
    logger.info("Test RMSE              : %.2f", test_metrics["rmse"])
    logger.info("Test MAE               : %.2f", test_metrics["mae"])
    logger.info("Test R2                : %.3f", test_metrics["r2"])
    logger.info("Test WAPE              : %.3f", test_metrics["wape"])
    logger.info("Outputs saved to %s/", output_dir)


if __name__ == "__main__":
    main()
