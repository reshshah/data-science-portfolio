import numpy as np
import pandas as pd

from src.routing import route_predict


class StubPipeline:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.probability), np.full(n, self.probability)])


def test_route_predict_splits_by_tenure_threshold():
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})
    tenure_days = pd.Series([10, 50, 200, 500])
    cold_start_pipe = StubPipeline(probability=0.9)
    primary_pipe = StubPipeline(probability=0.1)

    result = route_predict(cold_start_pipe, primary_pipe, X, tenure_days, tenure_threshold_days=90)

    assert result.tolist() == [0.9, 0.9, 0.1, 0.1]
