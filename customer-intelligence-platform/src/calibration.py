"""Calibration and ranking diagnostics for a binary classifier's probabilities.

ROC-AUC/PR-AUC measure ranking quality; they say nothing about whether a
predicted 0.7 means "70% of the time this actually happens." Calibration
answers that. The decile table answers the operational question a
retention/marketing team actually has: "if we act on the top N% by score,
how many of the real positives do we actually catch?"
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def compute_calibration(y, prob, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs. actual positive rate, per equal-size bin."""
    prob_true, prob_pred = calibration_curve(y, prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({
        "mean_predicted_probability": prob_pred,
        "actual_positive_rate": prob_true,
    })


def compute_decile_table(y, prob, n_deciles: int = 10) -> pd.DataFrame:
    """Rank observations into deciles by predicted probability.

    Decile 10 = highest predicted probability (highest risk for churn,
    highest propensity for propensity). Includes each decile's actual
    positive rate, lift over the overall base rate, and cumulative capture
    -- the numbers a lift chart and gains chart are drawn from.
    """
    df = pd.DataFrame({"y": np.asarray(y), "prob": np.asarray(prob)})

    # rank(method="first") breaks ties by position so qcut always produces
    # exactly n_deciles equal-sized groups, even with many repeated scores.
    ranked = df["prob"].rank(method="first")
    raw_bucket = pd.qcut(ranked, n_deciles, labels=False)
    df["decile"] = n_deciles - raw_bucket  # bucket 0 (lowest prob) -> decile n; highest prob -> decile 1...

    baseline_rate = df["y"].mean()
    total_positives = df["y"].sum()

    table = (
        df.groupby("decile")
        .agg(count=("y", "size"), positives=("y", "sum"), mean_probability=("prob", "mean"))
        .sort_index()  # decile 1 (highest prob) first
    )
    table["positive_rate"] = table["positives"] / table["count"]
    table["lift"] = table["positive_rate"] / baseline_rate
    table["cumulative_positives"] = table["positives"].cumsum()
    table["cumulative_capture_rate"] = table["cumulative_positives"] / total_positives
    table["cumulative_population_rate"] = table["count"].cumsum() / table["count"].sum()
    return table.reset_index()
