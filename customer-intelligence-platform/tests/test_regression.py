import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.evaluator import compute_regression_metrics
from src.plots import feature_importance
from src.preprocessing import build_preprocessor
from src.trainer import build_regression_estimator


def test_compute_regression_metrics_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    metrics = compute_regression_metrics(y, y)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["wape"] == 0.0


def test_compute_regression_metrics_wape_handles_zero_actuals():
    y = np.array([0.0, 0.0, 100.0])
    pred = np.array([10.0, 5.0, 90.0])

    metrics = compute_regression_metrics(y, pred)

    assert metrics["wape"] == (10 + 5 + 10) / 100


def test_build_regression_estimator_dispatches_by_type():
    ridge = build_regression_estimator({"type": "ridge", "alpha": 1.0}, random_state=42)
    assert ridge.__class__.__name__ == "Ridge"

    xgb = build_regression_estimator(
        {
            "type": "xgboost",
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        random_state=42,
    )
    assert xgb.__class__.__name__ == "XGBRegressor"

    lgbm = build_regression_estimator(
        {
            "type": "lightgbm",
            "n_estimators": 10,
            "num_leaves": 7,
            "max_depth": -1,
            "learning_rate": 0.1,
            "min_child_samples": 5,
        },
        random_state=42,
    )
    assert lgbm.__class__.__name__ == "LGBMRegressor"


def test_feature_importance_handles_1d_regression_coefficients():
    # Ridge.coef_ is 1D (n_features,), unlike LogisticRegression.coef_ which
    # is 2D (n_classes, n_features) — this would previously break on [0].
    df = pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0],
        "cat_a": ["x", "y", "x", "y"],
        "target": [10.0, 20.0, 15.0, 25.0],
    })
    pipe = Pipeline([
        ("preprocess", build_preprocessor(["num_a"], ["cat_a"])),
        ("model", Ridge(alpha=1.0, random_state=42)),
    ])
    pipe.fit(df[["num_a", "cat_a"]], df["target"])

    fi = feature_importance(pipe)

    assert len(fi) == 3
    assert set(fi["feature"]) == {"numeric__num_a", "categorical__cat_a_x", "categorical__cat_a_y"}
