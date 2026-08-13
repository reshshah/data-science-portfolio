"""Post-training model diagnostics: SHAP explainability, calibration, and
ranking (lift/gain/decile) analysis, bundled into one call so
train_classifier.py and tune_classifier.py don't duplicate it.
"""

from pathlib import Path

import pandas as pd

from src.calibration import compute_calibration, compute_decile_table
from src.explain import compute_shap_values, summarize_shap
from src.plots import (
    plot_calibration_curve,
    plot_gain_chart,
    plot_lift_chart,
    plot_shap_summary,
)


def run_diagnostics(
    pipe,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_prob,
    plot_dir: Path,
    metric_dir: Path,
) -> None:
    """Compute and save SHAP importance + calibration + decile/lift/gain,
    all evaluated on the held-out test set."""
    shap_values, X_transformed = compute_shap_values(pipe, X_test)
    plot_shap_summary(shap_values, X_transformed, plot_dir)
    summarize_shap(shap_values, list(X_transformed.columns)).to_csv(
        metric_dir / "shap_importance.csv", index=False
    )

    calibration_table = compute_calibration(y_test, test_prob)
    calibration_table.to_csv(metric_dir / "calibration_table.csv", index=False)
    plot_calibration_curve(calibration_table, plot_dir)

    decile_table = compute_decile_table(y_test, test_prob)
    decile_table.to_csv(metric_dir / "decile_table.csv", index=False)
    plot_lift_chart(decile_table, plot_dir)
    plot_gain_chart(decile_table, plot_dir)
