import numpy as np
import pandas as pd

from src.tuning import run_hyperparameter_search


def _synthetic_classification_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = pd.Series((x1 + 0.3 * x2 > 0).astype(int))
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y


def test_search_stays_within_the_configured_param_distributions():
    X, y = _synthetic_classification_data()
    config = {
        "random_state": 42,
        "model": {
            "type": "logistic_regression",
            "max_iter": 200,
            "class_weight": "balanced",
        },
        "tuning": {
            "cv_folds": 3,
            "n_iter": 4,
            "scoring": "roc_auc",
            "param_distributions": {"C": [0.01, 0.1, 1.0, 10.0]},
        },
    }

    search = run_hyperparameter_search(X, y, ["x1", "x2"], [], config)

    assert search.best_params_["model__C"] in [0.01, 0.1, 1.0, 10.0]
    assert len(search.cv_results_["mean_test_score"]) == 4
    # best_estimator_ is a fitted, drop-in-compatible Pipeline
    assert hasattr(search.best_estimator_, "predict_proba")
    preds = search.best_estimator_.predict_proba(X)
    assert preds.shape == (len(X), 2)


def test_search_respects_configured_n_iter():
    X, y = _synthetic_classification_data()
    config = {
        "random_state": 0,
        "model": {"type": "logistic_regression", "max_iter": 200, "class_weight": "balanced"},
        "tuning": {
            "cv_folds": 3,
            "n_iter": 2,
            "param_distributions": {"C": [0.1, 1.0, 10.0]},
        },
    }

    search = run_hyperparameter_search(X, y, ["x1", "x2"], [], config)

    assert len(search.cv_results_["mean_test_score"]) == 2
