import numpy as np
import pandas as pd
from econml.metalearners import XLearner
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge


def _build_outcome_model(model_cfg: dict, random_state: int):
    model_type = model_cfg["outcome_model"]
    if model_type == "ridge":
        return Ridge(alpha=model_cfg.get("ridge_alpha", 1.0), random_state=random_state)
    if model_type == "lightgbm":
        return LGBMRegressor(
            n_estimators=model_cfg["n_estimators"],
            num_leaves=model_cfg["num_leaves"],
            max_depth=model_cfg["max_depth"],
            learning_rate=model_cfg["learning_rate"],
            min_child_samples=model_cfg["min_child_samples"],
            random_state=random_state,
            verbose=-1,
        )
    raise ValueError(f"Unknown outcome model type: {model_type}")


def build_xlearner(model_cfg: dict, random_state: int) -> XLearner:
    """
    X-learner for churn uplift.

    Outcome models are regressors (not classifiers) fit on the binary
    churn label as a continuous 0/1 target -- econml's meta-learners call
    `.predict()`, not `.predict_proba()`, so a classifier would collapse
    the outcome to hard 0/1 labels and destroy the signal needed to
    estimate treatment effects.
    """
    outcome_model = _build_outcome_model(model_cfg, random_state)
    propensity_model = LogisticRegression(max_iter=1000, random_state=random_state)
    return XLearner(models=outcome_model, propensity_model=propensity_model)


def predict_uplift(estimator: XLearner, X) -> np.ndarray:
    """
    Predicted change in retention probability from treatment.

    econml's .effect() returns E[Y|T=1] - E[Y|T=0] on the outcome as given.
    Our outcome is churn (1 = bad), so we negate it: positive uplift here
    means treatment is predicted to REDUCE churn probability (persuadable);
    negative means treatment is predicted to INCREASE it (sleeping dog).
    """
    return -estimator.effect(X)


def uplift_by_bucket(y_true: pd.Series, treatment: pd.Series, predicted_uplift: np.ndarray, n_buckets: int = 4) -> pd.DataFrame:
    """
    Groups customers into buckets by predicted uplift (highest first) and
    compares realized uplift (control churn rate - treated churn rate)
    against the model's average prediction in that bucket. A model with
    real signal should show realized uplift decreasing as predicted uplift
    decreases across buckets.
    """
    df = pd.DataFrame({
        "y": y_true.to_numpy(),
        "treatment": treatment.to_numpy(),
        "predicted_uplift": predicted_uplift,
    })
    df["bucket"] = pd.qcut(df["predicted_uplift"].rank(method="first", ascending=False), n_buckets, labels=False) + 1

    rows = []
    for bucket, group in df.groupby("bucket"):
        treated = group.loc[group["treatment"] == 1, "y"]
        control = group.loc[group["treatment"] == 0, "y"]
        rows.append({
            "bucket": bucket,
            "n": len(group),
            "n_treated": len(treated),
            "n_control": len(control),
            "mean_predicted_uplift": group["predicted_uplift"].mean(),
            "treated_churn_rate": treated.mean() if len(treated) else np.nan,
            "control_churn_rate": control.mean() if len(control) else np.nan,
            "realized_uplift": (control.mean() - treated.mean()) if len(treated) and len(control) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("bucket")
