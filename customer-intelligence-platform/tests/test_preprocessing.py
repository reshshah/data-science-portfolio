import pandas as pd
import pytest

from src.feature_validation import FeatureValidationError, validate_features
from src.preprocessing import build_preprocessor, split_xy


def make_df():
    return pd.DataFrame({
        "num_a": [1.0, 2.0, None],
        "cat_a": ["x", "y", "x"],
        "target": [0, 1, 0],
    })


def test_split_xy_selects_expected_columns():
    df = make_df()
    X, y = split_xy(df, "target", ["num_a"], ["cat_a"])
    assert list(X.columns) == ["num_a", "cat_a"]
    assert y.tolist() == [0, 1, 0]


def test_build_preprocessor_handles_missing_and_categorical():
    df = make_df()
    X, y = split_xy(df, "target", ["num_a"], ["cat_a"])
    preprocessor = build_preprocessor(["num_a"], ["cat_a"])
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == 3
    assert not pd.isna(transformed).any()


def test_validate_features_raises_on_missing_column():
    df = pd.DataFrame({"num_a": [1.0], "target": [0]})
    metadata = {
        "target": "target",
        "numeric_features": ["num_a", "num_b"],
        "categorical_features": [],
    }
    with pytest.raises(FeatureValidationError):
        validate_features(df, metadata)


def test_validate_features_passes_with_expected_columns():
    df = pd.DataFrame({"num_a": [1.0], "cat_a": ["x"], "target": [0]})
    metadata = {
        "target": "target",
        "numeric_features": ["num_a"],
        "categorical_features": ["cat_a"],
    }
    validate_features(df, metadata)
