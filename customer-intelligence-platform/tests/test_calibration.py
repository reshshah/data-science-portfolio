import numpy as np
import pandas as pd
import pytest

from src.calibration import compute_calibration, compute_decile_table


def test_decile_table_covers_every_row_exactly_once():
    y = pd.Series([1] * 20 + [0] * 80)
    prob = pd.Series(np.arange(100) / 100.0)

    table = compute_decile_table(y, prob, n_deciles=10)

    assert list(table["decile"]) == list(range(1, 11))
    assert table["count"].sum() == 100
    assert table["positives"].sum() == 20  # every positive accounted for, none double-counted


def test_decile_table_lift_is_monotonically_non_increasing_for_perfect_ranking():
    # Probability perfectly ranks the positives to the top -- lift should
    # strictly decrease decile over decile.
    y = pd.Series([1] * 20 + [0] * 80)
    prob = pd.Series(np.linspace(1.0, 0.0, 100))  # highest prob first, matching y's order

    table = compute_decile_table(y, prob, n_deciles=10)

    assert table["decile"].iloc[0] == 1
    assert table.loc[table["decile"] == 1, "positive_rate"].iloc[0] == 1.0
    assert table.loc[table["decile"] == 10, "positive_rate"].iloc[0] == 0.0
    lifts = table.sort_values("decile")["lift"].tolist()
    assert lifts == sorted(lifts, reverse=True)


def test_decile_table_cumulative_columns_reach_one():
    rng = np.random.default_rng(7)
    y = pd.Series(rng.integers(0, 2, size=500))
    prob = pd.Series(rng.random(500))

    table = compute_decile_table(y, prob, n_deciles=10)

    assert table["cumulative_capture_rate"].iloc[-1] == pytest.approx(1.0)
    assert table["cumulative_population_rate"].iloc[-1] == pytest.approx(1.0)


def test_calibration_returns_one_row_per_bin_within_valid_range():
    # Low-probability group mostly negative, high-probability group mostly
    # positive -- two quantile bins, each with a sensible actual rate.
    prob = pd.Series([0.1] * 5 + [0.9] * 5)
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])

    calib = compute_calibration(y, prob, n_bins=2)

    assert len(calib) == 2
    assert set(calib.columns) == {"mean_predicted_probability", "actual_positive_rate"}
    assert calib["actual_positive_rate"].between(0, 1).all()
    assert calib["actual_positive_rate"].iloc[0] < calib["actual_positive_rate"].iloc[1]
