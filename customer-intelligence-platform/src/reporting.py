import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_json(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_predictions(y, prob, pred, path: Path) -> None:
    pd.DataFrame({"actual": y, "probability": prob, "prediction": pred}).to_csv(path, index=False)


def print_summary(dummy_metrics: dict | None, val_metrics: dict, test_metrics: dict, threshold: float, output_dir: Path, title: str = "Training Complete") -> None:
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    if dummy_metrics is not None:
        logger.info("Dummy Validation ROC-AUC : %.3f", dummy_metrics["roc_auc"])
    logger.info("Validation ROC-AUC        : %.3f", val_metrics["roc_auc"])
    logger.info("Validation PR-AUC         : %.3f", val_metrics["pr_auc"])
    logger.info("Best Threshold            : %.2f", threshold)
    logger.info("Test ROC-AUC              : %.3f", test_metrics["roc_auc"])
    logger.info("Test PR-AUC               : %.3f", test_metrics["pr_auc"])
    logger.info("Outputs saved to %s/", output_dir)
