#!/usr/bin/env python3
"""Cross-validated hyperparameter search for a binary classifier, from a
config file with a "tuning" section on top of the usual "model" section.

The search happens entirely within train (StratifiedKFold), never touching
validation or test. The winning pipeline is then refit on the full
training set and evaluated exactly the same way models/train_classifier.py
evaluates a manually-configured model, so results are directly comparable.

Run:
    python3 models/tune_classifier.py --config configs/churn_lightgbm_tuned_config.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_feature_metadata, load_split
from src.diagnostics import run_diagnostics
from src.evaluator import best_threshold, compute_metrics
from src.feature_validation import validate_features
from src.plots import feature_importance, plot_feature_importance, plot_roc_pr
from src.preprocessing import split_xy
from src.reporting import print_summary, save_json, save_predictions
from src.tracking import log_config_params, log_metrics, start_run
from src.trainer import train_dummy
from src.tuning import run_hyperparameter_search
from src.utils import get_project_root, load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validated hyperparameter search for a binary classifier."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a config with a 'tuning' section (relative to project root).",
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

    with start_run(config, root, run_name=f"{model_type}_tuned"):
        log_config_params(config)
        tuning_cfg = config["tuning"]
        mlflow.log_params({
            "tuning__cv_folds": tuning_cfg.get("cv_folds", 5),
            "tuning__n_iter": tuning_cfg.get("n_iter", 20),
            "tuning__scoring": tuning_cfg.get("scoring", "roc_auc"),
        })

        logger.info(
            "Searching %d hyperparameter combinations, %d-fold CV, scoring=%s",
            tuning_cfg.get("n_iter", 20),
            tuning_cfg.get("cv_folds", 5),
            tuning_cfg.get("scoring", "roc_auc"),
        )
        search = run_hyperparameter_search(
            X_train, y_train, numeric_features, categorical_features, config
        )
        pipe = search.best_estimator_

        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        cv_results.to_csv(metric_dir / "cv_results.csv", index=False)
        save_json(
            {"best_cv_score": float(search.best_score_), "best_params": search.best_params_},
            metric_dir / "best_params.json",
        )
        mlflow.log_metric("best_cv_score", float(search.best_score_))
        mlflow.log_params({
            f"best__{k.replace('model__', '')}": v for k, v in search.best_params_.items()
        })
        logger.info("Best CV %s: %.4f | params: %s", tuning_cfg.get("scoring", "roc_auc"), search.best_score_, search.best_params_)

        dummy = train_dummy(X_train, y_train)
        dummy_prob = dummy.predict_proba(X_valid)[:, 1]
        dummy_metrics = compute_metrics(y_valid, dummy.predict(X_valid), dummy_prob)
        save_json(dummy_metrics, metric_dir / "dummy_validation.json")
        log_metrics("dummy_validation", dummy_metrics)

        val_prob = pipe.predict_proba(X_valid)[:, 1]
        threshold, _ = best_threshold(y_valid, val_prob, config)
        val_pred = (val_prob >= threshold).astype(int)
        val_metrics = compute_metrics(y_valid, val_pred, val_prob)
        val_metrics["threshold"] = threshold
        save_json(val_metrics, metric_dir / "validation_metrics.json")
        save_predictions(y_valid, val_prob, val_pred, pred_dir / "validation_predictions.csv")
        log_metrics("validation", val_metrics)

        test_prob = pipe.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        test_metrics = compute_metrics(y_test, test_pred, test_prob)
        test_metrics["threshold"] = threshold
        save_json(test_metrics, metric_dir / "test_metrics.json")
        save_predictions(y_test, test_prob, test_pred, pred_dir / "test_predictions.csv")
        log_metrics("test", test_metrics)

        joblib.dump(pipe, model_dir / f"{model_type}.pkl")
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="pickle")

        fi = feature_importance(pipe)
        fi.to_csv(metric_dir / "feature_importance.csv", index=False)
        plot_feature_importance(fi, plot_dir)
        plot_roc_pr(y_test, test_prob, plot_dir)

        run_diagnostics(pipe, X_test, y_test, test_prob, plot_dir, metric_dir)

        for artifact in plot_dir.glob("*.png"):
            mlflow.log_artifact(str(artifact), artifact_path="plots")
        for artifact in metric_dir.glob("*.csv"):
            mlflow.log_artifact(str(artifact), artifact_path="metrics")

        print_summary(dummy_metrics, val_metrics, test_metrics, threshold, output_dir, title="Tuning Complete")
        logger.info("Best params: %s", search.best_params_)


if __name__ == "__main__":
    main()
