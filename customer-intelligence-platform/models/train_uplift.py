#!/usr/bin/env python3
"""
Trains an X-learner to estimate the effect of marketing outreach on churn.

Unlike the classifiers/regressors elsewhere in this project, this script
doesn't use src/trainer.py's Pipeline pattern directly -- econml's XLearner
needs the treatment column (T) and outcome (Y) passed separately from the
covariates (X), so preprocessing is done once upfront instead of bundled
into the estimator.

See docs/MODEL_METRICS_NOTES.md for what "uplift" means here and how to
read the bucket table this script prints.
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_feature_metadata, load_split
from src.preprocessing import build_preprocessor
from src.uplift import build_xlearner, predict_uplift, uplift_by_bucket
from src.utils import get_project_root, load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the churn uplift (X-learner) model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/uplift_config.yaml"),
        help="Path to the uplift model config YAML (relative to project root).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = get_project_root()
    config = load_config(root / args.config)
    setup_logging(root / config["logging"]["file"], config["logging"]["level"])

    data_dir = root / config["paths"]["data_dir"]
    metadata = load_feature_metadata(root / config["paths"]["metadata_file"])
    treatment_col = metadata["treatment_column"]
    outcome_col = metadata["outcome_column"]
    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    output_dir = root / config["paths"]["output_dir"] / config["name"]
    model_dir = output_dir / "models"
    metric_dir = output_dir / "metrics"
    for d in (model_dir, metric_dir):
        d.mkdir(parents=True, exist_ok=True)

    train = load_split(data_dir, "train")
    valid = load_split(data_dir, "validation")
    test = load_split(data_dir, "test")

    feature_cols = numeric_features + categorical_features
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train = preprocessor.fit_transform(train[feature_cols])
    X_valid = preprocessor.transform(valid[feature_cols])
    X_test = preprocessor.transform(test[feature_cols])

    T_train, Y_train = train[treatment_col].to_numpy(), train[outcome_col].to_numpy()

    logger.info(
        "Fitting X-learner: %d train rows (%d treatment, %d control)",
        len(train), T_train.sum(), (T_train == 0).sum(),
    )
    estimator = build_xlearner(config["model"], config["random_state"])
    estimator.fit(Y_train, T_train, X=X_train)

    valid_uplift = predict_uplift(estimator, X_valid)
    test_uplift = predict_uplift(estimator, X_test)

    n_buckets = config["evaluation"]["n_buckets"]
    valid_buckets = uplift_by_bucket(valid[outcome_col], valid[treatment_col], valid_uplift, n_buckets)
    test_buckets = uplift_by_bucket(test[outcome_col], test[treatment_col], test_uplift, n_buckets)

    valid_buckets.to_csv(metric_dir / "validation_uplift_by_bucket.csv", index=False)
    test_buckets.to_csv(metric_dir / "test_uplift_by_bucket.csv", index=False)

    joblib.dump({"preprocessor": preprocessor, "estimator": estimator}, model_dir / "xlearner.pkl")

    logger.info("=" * 60)
    logger.info("Uplift Training Complete")
    logger.info("=" * 60)
    logger.info("Mean predicted uplift (validation): %.4f", valid_uplift.mean())
    logger.info("Mean predicted uplift (test):        %.4f", test_uplift.mean())
    logger.info("\nValidation uplift by bucket (1 = highest predicted uplift):\n%s", valid_buckets.to_string(index=False))
    logger.info("\nTest uplift by bucket (1 = highest predicted uplift):\n%s", test_buckets.to_string(index=False))
    logger.info("Outputs saved to %s/", output_dir)


if __name__ == "__main__":
    main()
