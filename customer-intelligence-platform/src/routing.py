import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def route_predict(
    cold_start_pipe: Pipeline,
    primary_pipe: Pipeline,
    X: pd.DataFrame,
    tenure_days: pd.Series,
    tenure_threshold_days: int,
) -> np.ndarray:
    """Blend two already-trained pipelines by customer tenure.

    Customers at or below the tenure threshold are scored by cold_start_pipe
    (the model that generalizes better to customers with little history);
    everyone else is scored by primary_pipe (the stronger model overall).
    """
    primary_prob = primary_pipe.predict_proba(X)[:, 1]
    cold_start_prob = cold_start_pipe.predict_proba(X)[:, 1]
    use_cold_start = (tenure_days <= tenure_threshold_days).to_numpy()
    return np.where(use_cold_start, cold_start_prob, primary_prob)
