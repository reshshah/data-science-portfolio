"""Tests for point-in-time feature construction.

The critical test here is test_no_future_leakage: it proves that transactions
after the snapshot date cannot influence features. That single property is
what separates a trustworthy training set from an untrustworthy one.

Run from the customer-intelligence-platform directory: pytest tests/ -v
"""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipelines.build_retail_features import (
    PROPENSITY_TARGET,
    RETAIL_FEATURES,
    TARGET,
    TARGET_COLUMNS,
    build_multiple_snapshots,
    build_snapshot,
    load_transactions,
)

SNAPSHOT = "2011-01-01"


def make_txn(customer, invoice, date, qty=1, price=10.0, stock="A1"):
    return {
        "invoice": invoice, "stock_code": stock, "quantity": qty,
        "invoice_date": pd.Timestamp(date), "unit_price": price,
        "customer_id": customer, "is_return": str(invoice).startswith("C"),
        "line_revenue": qty * price,
    }


@pytest.fixture
def txns():
    """Three customers with deliberately different behavior."""
    rows = [
        # loyal: buys before and after the snapshot → not churned
        make_txn("1", "100", "2010-06-01"),
        make_txn("1", "101", "2010-12-15", qty=2),
        make_txn("1", "102", "2011-02-01"),
        # lapsing: bought long ago, nothing in the 90d window, never returns → churned
        make_txn("2", "200", "2010-02-01"),
        # returner: buys in window but also cancels, no repurchase → churned
        make_txn("3", "300", "2010-12-01", qty=5),
        make_txn("3", "C301", "2010-12-10", qty=-2),
    ]
    return pd.DataFrame(rows)


def test_labels_reflect_future_window(txns):
    out = build_snapshot(txns, SNAPSHOT).set_index("customer_id")
    assert out.loc["1", TARGET] == 0  # repurchased after snapshot
    assert out.loc["2", TARGET] == 1
    assert out.loc["3", TARGET] == 1


def test_propensity_reflects_short_window(txns):
    """propensity_label_30d and churn_label_180d measure different things:
    customer 1 repurchases 31 days after the snapshot -- inside the 180-day
    churn window (not churned) but outside the 30-day propensity window
    (no near-term propensity)."""
    out = build_snapshot(txns, SNAPSHOT).set_index("customer_id")
    assert out.loc["1", TARGET] == 0
    assert out.loc["1", PROPENSITY_TARGET] == 0
    assert out.loc["2", PROPENSITY_TARGET] == 0
    assert out.loc["3", PROPENSITY_TARGET] == 0

    near_term = pd.concat([
        txns, pd.DataFrame([make_txn("2", "900", "2011-01-15")])
    ], ignore_index=True)
    out2 = build_snapshot(near_term, SNAPSHOT).set_index("customer_id")
    assert out2.loc["2", PROPENSITY_TARGET] == 1


def test_recency_and_tenure(txns):
    out = build_snapshot(txns, SNAPSHOT).set_index("customer_id")
    # customer 1 last bought 2010-12-15 → 17 days before 2011-01-01
    assert out.loc["1", "recency_days"] == 17
    # first purchase 2010-06-01 → 214 days of tenure
    assert out.loc["1", "tenure_days"] == 214


def test_window_features_exclude_old_purchases(txns):
    out = build_snapshot(txns, SNAPSHOT).set_index("customer_id")
    # customer 2's only purchase (Feb 2010) is outside the 90-day window
    assert out.loc["2", "frequency_90d"] == 0
    assert out.loc["2", "monetary_90d"] == 0
    assert out.loc["2", "avg_basket_value_90d"] == 0


def test_return_rate(txns):
    out = build_snapshot(txns, SNAPSHOT).set_index("customer_id")
    assert out.loc["3", "return_rate_90d"] == pytest.approx(0.5)  # 1 of 2 lines
    assert out.loc["1", "return_rate_90d"] == 0.0


def test_no_future_leakage(txns):
    """Adding huge purchases AFTER the snapshot must not change any feature."""
    before = build_snapshot(txns, SNAPSHOT).set_index("customer_id")[RETAIL_FEATURES]

    polluted = pd.concat([
        txns,
        pd.DataFrame([
            make_txn("1", "999", "2011-06-01", qty=1000, price=500.0),
            make_txn("2", "998", "2011-07-01", qty=1000, price=500.0),
        ]),
    ], ignore_index=True)
    after = build_snapshot(polluted, SNAPSHOT).set_index("customer_id")[RETAIL_FEATURES]

    pd.testing.assert_frame_equal(before, after)


def test_customers_without_history_are_excluded(txns):
    """A customer whose first purchase is after the snapshot can't be scored."""
    with_newcomer = pd.concat([
        txns, pd.DataFrame([make_txn("99", "500", "2011-03-01")])
    ], ignore_index=True)
    out = build_snapshot(with_newcomer, SNAPSHOT)
    assert "99" not in set(out["customer_id"])


def test_multiple_snapshots_stack(txns):
    out = build_multiple_snapshots(txns, ["2010-12-01", "2011-01-01"])
    assert set(out["snapshot_date"]) == {"2010-12-01", "2011-01-01"}
    assert len(out) > len(build_snapshot(txns, "2011-01-01"))


def test_schema_is_stable(txns):
    out = build_snapshot(txns, SNAPSHOT)
    assert list(out.columns) == ["customer_id", "snapshot_date"] + RETAIL_FEATURES + TARGET_COLUMNS
    assert out[RETAIL_FEATURES].notna().all().all(), "features must never be NaN"
    assert out[TARGET_COLUMNS].notna().all().all(), "labels must never be NaN"


def test_load_transactions_cleans(tmp_path):
    raw = pd.DataFrame([
        {"Invoice": "1", "StockCode": "A", "Quantity": 1, "InvoiceDate": "2010-01-01",
         "Price": 5.0, "Customer ID": 111.0},
        {"Invoice": "2", "StockCode": "B", "Quantity": 1, "InvoiceDate": "2010-01-02",
         "Price": 5.0, "Customer ID": None},          # no customer → dropped
        {"Invoice": "3", "StockCode": "C", "Quantity": 1, "InvoiceDate": "2010-01-03",
         "Price": 0.0, "Customer ID": 222.0},         # bad price → dropped
        {"Invoice": "C4", "StockCode": "D", "Quantity": -1, "InvoiceDate": "2010-01-04",
         "Price": 5.0, "Customer ID": 111.0},         # cancellation → kept, flagged
    ])
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)

    df = load_transactions(path)
    assert len(df) == 2
    assert set(df["customer_id"]) == {"111"}
    assert df["is_return"].sum() == 1