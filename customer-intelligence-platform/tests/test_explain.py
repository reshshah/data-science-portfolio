import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.explain import compute_shap_values, summarize_shap
from src.preprocessing import build_preprocessor


def _fit_logistic_pipeline():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # target driven almost entirely by x1 -- x2 should end up with much
    # smaller SHAP importance.
    y = (x1 + 0.05 * x2 > 0).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2})

    pipe = Pipeline([
        ("preprocess", build_preprocessor(["x1", "x2"], [])),
        ("model", LogisticRegression(random_state=0)),
    ])
    pipe.fit(df, y)
    return pipe, df


def test_compute_shap_values_shape_matches_sample():
    pipe, X = _fit_logistic_pipeline()

    shap_values, X_transformed = compute_shap_values(pipe, X, sample_size=50)

    assert shap_values.shape == (50, 2)
    assert list(X_transformed.columns) == ["x1", "x2"]


def test_compute_shap_values_strips_column_transformer_prefix():
    pipe, X = _fit_logistic_pipeline()

    _, X_transformed = compute_shap_values(pipe, X, sample_size=20)

    assert "numeric__x1" not in X_transformed.columns
    assert "x1" in X_transformed.columns


def test_summarize_shap_ranks_the_dominant_feature_first():
    pipe, X = _fit_logistic_pipeline()
    shap_values, X_transformed = compute_shap_values(pipe, X, sample_size=200)

    summary = summarize_shap(shap_values, list(X_transformed.columns))

    assert summary.iloc[0]["feature"] == "x1"
    assert summary["mean_abs_shap"].is_monotonic_decreasing
