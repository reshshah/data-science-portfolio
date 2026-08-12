import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureValidationError(Exception):
    pass


def validate_features(df: pd.DataFrame, metadata: dict) -> None:
    expected = (
        metadata["numeric_features"]
        + metadata["categorical_features"]
        + metadata.get("boolean_features", [])
    )
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise FeatureValidationError(f"Missing expected feature columns: {missing}")

    target = metadata["target"]
    if target not in df.columns:
        raise FeatureValidationError(f"Missing target column: {target}")

    null_counts = df[expected].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        logger.warning("Unexpected nulls found in features:\n%s", bad)

    logger.info(
        "Feature validation passed: %d numeric, %d categorical",
        len(metadata["numeric_features"]),
        len(metadata["categorical_features"]),
    )
