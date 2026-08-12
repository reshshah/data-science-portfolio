import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def compute_metrics(y, pred, prob) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def compute_regression_metrics(y, pred) -> dict:
    y = np.asarray(y)
    pred = np.asarray(pred)
    return {
        "rmse": float(root_mean_squared_error(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        # weighted absolute percentage error: safe with a zero-inflated
        # target (~25% of customers have future_revenue_180d == 0), unlike
        # standard MAPE which divides by each individual actual value.
        "wape": float(np.abs(y - pred).sum() / np.abs(y).sum()),
    }


def best_threshold(y, prob, config: dict) -> tuple[float, float]:
    t_cfg = config["threshold"]
    best_t, best_score = 0.5, -1.0
    for t in np.arange(t_cfg["search_min"], t_cfg["search_max"], t_cfg["search_step"]):
        pred = (prob >= t).astype(int)
        score = f1_score(y, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score
