 """Unit tests for the pure calculation functions in marketing-analytics/.

Run with: pytest
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# marketing-analytics has a hyphen, so add it to the path for imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "marketing-analytics"))

from ads_performance import add_ctr_cpc
from breakeven_roas import breakeven_roas, project_revenue
from mmm_assistant import CHANNELS, TARGET, allocate_budget, fit_lasso


# --- breakeven_roas ---

def test_breakeven_roas_basic():
    assert breakeven_roas(0.40) == 2.5
    assert breakeven_roas(0.50) == 2.0
    assert breakeven_roas(1.0) == 1.0


def test_breakeven_roas_rejects_bad_margin():
    with pytest.raises(ValueError):
        breakeven_roas(0)
    with pytest.raises(ValueError):
        breakeven_roas(1.5)


def test_project_revenue():
    assert project_revenue([100, 200], [3.0, 1.5]) == [300.0, 300.0]


# --- ads_performance ---

def test_add_ctr_cpc_normal_rows():
    df = pd.DataFrame({"clicks": [10], "impressions": [1000], "cost": [5.0]})
    out = add_ctr_cpc(df)
    assert out["CTR"].iloc[0] == 1.0
    assert out["CPC"].iloc[0] == 0.5


def test_add_ctr_cpc_zero_denominators():
    df = pd.DataFrame({"clicks": [0], "impressions": [0], "cost": [5.0]})
    out = add_ctr_cpc(df)
    assert np.isnan(out["CTR"].iloc[0])
    assert np.isnan(out["CPC"].iloc[0])


# --- mmm_assistant ---

def test_allocate_budget_sums_to_budget():
    coefs = {"TV_spend": 2.0, "Digital_spend": 1.0, "Social_spend": 1.0, "Search_spend": 0.0}
    alloc = allocate_budget(coefs, 1000)
    assert sum(alloc.values()) == pytest.approx(1000)
    assert alloc["TV_spend"] == pytest.approx(500)
    assert alloc["Search_spend"] == 0


def test_allocate_budget_ignores_negative_coefficients():
    coefs = {"TV_spend": 3.0, "Digital_spend": -5.0, "Social_spend": 1.0, "Search_spend": 0.0}
    alloc = allocate_budget(coefs, 800)
    assert alloc["Digital_spend"] == 0
    assert sum(alloc.values()) == pytest.approx(800)


def test_allocate_budget_all_zero_coefficients():
    coefs = {ch: 0.0 for ch in CHANNELS}
    assert all(v == 0 for v in allocate_budget(coefs, 500).values())


def test_fit_lasso_recovers_dominant_channel():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({ch: rng.uniform(100, 1000, n) for ch in CHANNELS})
    # Sales driven overwhelmingly by TV spend
    df[TARGET] = 5.0 * df["TV_spend"] + 0.1 * df["Search_spend"] + rng.normal(0, 50, n)
    _, coefs = fit_lasso(df)
    assert max(coefs, key=lambda ch: abs(coefs[ch])) == "TV_spend"
    assert coefs["TV_spend"] > 0
