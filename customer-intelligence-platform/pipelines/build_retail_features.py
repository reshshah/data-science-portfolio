"""
build_retail_features.py

Builds leakage-safe, point-in-time customer snapshots from the UCI Online
Retail II transaction log:

    features  <- computed ONLY from transactions strictly BEFORE snapshot_date
    labels    <- computed ONLY from transactions AFTER (and including)
                 snapshot_date, each over its own forward window

Two labels are produced from the same snapshot, at two different horizons:
    churn_label_180d      1 = no purchase in the 180 days after the snapshot
    propensity_label_30d  1 = a purchase in the 30 days after the snapshot

They deliberately measure different things over different windows (a
customer can be "not churned" over 180 days while still having low 30-day
propensity), so each downstream model must drop the OTHER label from its
feature set -- both are derived from the same future data, so using one to
predict the other is label leakage, not a legitimate feature.

Expected input:
    data/raw/online_retail_II.csv

Default output:
    data/processed/retail_training_snapshots.csv

Run from the project root:
    python3 pipelines/build_retail_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("data/raw/online_retail_II.csv")
DEFAULT_OUTPUT = Path("data/processed/retail_training_snapshots.csv")

LOOKBACK_WINDOW_DAYS = 90
LABEL_HORIZON_DAYS = 180
PROPENSITY_HORIZON_DAYS = 30

RETAIL_FEATURES = [
    "recency_days",
    "frequency_90d",
    "monetary_90d",
    "tenure_days",
    "distinct_products_90d",
    "avg_basket_value_90d",
    "return_rate_90d",
]
TARGET = "churn_label_180d"
PROPENSITY_TARGET = "propensity_label_30d"
TARGET_COLUMNS = [TARGET, PROPENSITY_TARGET]


def load_transactions(path: Path) -> pd.DataFrame:
    """Load and clean raw Online Retail II transactions.

    Cleaning decisions:
      - Rows without a Customer ID can't be attributed to a customer -> dropped.
      - Invoices starting with 'C' are cancellations -> kept, flagged as returns
        (they carry real signal: returns predict churn).
      - Non-positive unit prices are data errors -> dropped.
    """
    df = pd.read_csv(path, parse_dates=["InvoiceDate"], low_memory=False)

    # The two sheets use slightly different column spellings across releases.
    df = df.rename(columns={
        "Customer ID": "customer_id", "CustomerID": "customer_id",
        "Invoice": "invoice", "InvoiceNo": "invoice",
        "Price": "unit_price", "UnitPrice": "unit_price",
        "InvoiceDate": "invoice_date", "Quantity": "quantity",
        "StockCode": "stock_code",
    })

    df = df.dropna(subset=["customer_id"])
    df["customer_id"] = df["customer_id"].astype(int).astype(str)
    df["is_return"] = df["invoice"].astype(str).str.startswith("C")
    df = df[df["unit_price"] > 0]
    df["line_revenue"] = df["quantity"] * df["unit_price"]
    return df


def build_snapshot(
    txns: pd.DataFrame,
    snapshot_date,
    label_horizon_days: int = LABEL_HORIZON_DAYS,
) -> pd.DataFrame:
    """Features from history before snapshot_date; label from the window after.

    Only customers with at least one purchase before the snapshot are
    eligible -- you cannot score a customer who does not yet exist.
    """
    snapshot_date = pd.Timestamp(snapshot_date)
    label_end = snapshot_date + pd.Timedelta(days=label_horizon_days)
    propensity_end = snapshot_date + pd.Timedelta(days=PROPENSITY_HORIZON_DAYS)

    past = txns[txns["invoice_date"] < snapshot_date]
    future = txns[
        (txns["invoice_date"] >= snapshot_date) & (txns["invoice_date"] < label_end)
    ]
    future_30d = txns[
        (txns["invoice_date"] >= snapshot_date) & (txns["invoice_date"] < propensity_end)
    ]
    if past.empty:
        raise ValueError(f"No transactions before {snapshot_date.date()}")

    window_90 = past[past["invoice_date"] >= snapshot_date - pd.Timedelta(days=LOOKBACK_WINDOW_DAYS)]
    purchases_90 = window_90[~window_90["is_return"]]

    # --- lifetime features ---
    lifetime = past.groupby("customer_id").agg(
        last_purchase=("invoice_date", "max"),
        first_purchase=("invoice_date", "min"),
    )
    feat = pd.DataFrame(index=lifetime.index)
    feat["recency_days"] = (snapshot_date - lifetime["last_purchase"]).dt.days
    feat["tenure_days"] = (snapshot_date - lifetime["first_purchase"]).dt.days

    # --- 90-day window features ---
    g = purchases_90.groupby("customer_id")
    feat["frequency_90d"] = g["invoice"].nunique()
    feat["monetary_90d"] = g["line_revenue"].sum()
    feat["distinct_products_90d"] = g["stock_code"].nunique()

    # Customers with no activity in the window get true zeros, not NaN.
    for col in ["frequency_90d", "monetary_90d", "distinct_products_90d"]:
        feat[col] = feat[col].fillna(0)

    feat["avg_basket_value_90d"] = np.where(
        feat["frequency_90d"] > 0, feat["monetary_90d"] / feat["frequency_90d"], 0.0
    )

    # Return rate: returned line items as a share of all line items in the window.
    total_lines = window_90.groupby("customer_id").size()
    return_lines = window_90[window_90["is_return"]].groupby("customer_id").size()
    feat["return_rate_90d"] = (
        return_lines.reindex(feat.index).fillna(0) / total_lines.reindex(feat.index)
    ).fillna(0.0)

    # --- churn label: no purchase in the 180-day forward window ---
    repurchased = set(future[~future["is_return"]]["customer_id"])
    feat[TARGET] = (~feat.index.isin(repurchased)).astype(int)

    # --- propensity label: a purchase in the (shorter) 30-day forward window ---
    repurchased_30d = set(future_30d[~future_30d["is_return"]]["customer_id"])
    feat[PROPENSITY_TARGET] = feat.index.isin(repurchased_30d).astype(int)

    feat = feat.reset_index()
    feat.insert(1, "snapshot_date", snapshot_date.date().isoformat())
    return feat[["customer_id", "snapshot_date"] + RETAIL_FEATURES + TARGET_COLUMNS]


def build_multiple_snapshots(
    txns: pd.DataFrame,
    snapshot_dates: list[str],
    **kwargs,
) -> pd.DataFrame:
    """Stack several point-in-time snapshots into one training table.

    Multiple snapshots per customer is how you get enough rows from a modest
    customer base -- and it forces the model to learn behavior patterns
    rather than memorize a single point in time.
    """
    return pd.concat(
        [build_snapshot(txns, date, **kwargs) for date in snapshot_dates],
        ignore_index=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe monthly retail training snapshots."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--start-date",
        default="2010-03-01",
        help="First monthly snapshot date.",
    )
    parser.add_argument(
        "--end-date",
        default="2011-06-01",
        help="Last monthly snapshot date.",
    )
    return parser.parse_args()


def create_snapshot_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Create one snapshot at the beginning of every month."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("--start-date must not be after --end-date.")

    return pd.date_range(start=start, end=end, freq="MS")


def validate_training_dataset(training_data: pd.DataFrame) -> None:
    """Validate the final customer-snapshot grain and label."""
    print("\nValidating final training dataset")

    duplicate_rows = training_data.duplicated(
        subset=["customer_id", "snapshot_date"]
    ).sum()
    if duplicate_rows:
        raise ValueError(f"Found {duplicate_rows} duplicate customer-snapshot rows.")

    if training_data[RETAIL_FEATURES + TARGET_COLUMNS].isna().sum().sum() > 0:
        raise ValueError("Training dataset contains missing values.")

    for target in TARGET_COLUMNS:
        invalid = ~training_data[target].isin([0, 1])
        if invalid.any():
            raise ValueError(f"{target} contains values other than 0 and 1.")

    print(
        f"  Rows: {len(training_data):,}\n"
        f"  Unique customers: {training_data['customer_id'].nunique():,}\n"
        f"  Snapshot dates: {training_data['snapshot_date'].nunique():,}\n"
        f"  Churn rate: {training_data[TARGET].mean():.2%}\n"
        f"  Propensity rate: {training_data[PROPENSITY_TARGET].mean():.2%}"
    )


def main() -> None:
    args = parse_arguments()

    print("=" * 72)
    print("RETAIL (ONLINE RETAIL II) TRAINING SNAPSHOT PIPELINE")
    print("=" * 72)

    print(f"\nLoading transactions from {args.input.resolve()}")
    txns = load_transactions(args.input)
    print(
        f"  Kept {len(txns):,} rows | "
        f"{txns['customer_id'].nunique():,} customers | "
        f"{txns['invoice_date'].min().date()} to "
        f"{txns['invoice_date'].max().date()}"
    )

    snapshot_dates = create_snapshot_dates(args.start_date, args.end_date)
    maximum_invoice_date = txns["invoice_date"].max()

    all_snapshots = []
    for snapshot_date in snapshot_dates:
        required_end_date = snapshot_date + pd.Timedelta(days=LABEL_HORIZON_DAYS)
        if required_end_date > maximum_invoice_date + pd.Timedelta(days=1):
            print(
                f"\nSkipping {snapshot_date.date()}: "
                "insufficient future data for the 180-day label."
            )
            continue

        snapshot = build_snapshot(txns, str(snapshot_date.date()))
        all_snapshots.append(snapshot)
        print(
            f"\n{snapshot_date.date()}: {len(snapshot):,} rows | "
            f"churn rate={snapshot[TARGET].mean():.2%} | "
            f"propensity rate={snapshot[PROPENSITY_TARGET].mean():.2%}"
        )

    if not all_snapshots:
        raise ValueError(
            "No snapshots were created. Check the requested dates and "
            "the available transaction date range."
        )

    training_data = pd.concat(all_snapshots, ignore_index=True)
    training_data = training_data.sort_values(
        ["snapshot_date", "customer_id"]
    ).reset_index(drop=True)

    validate_training_dataset(training_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(args.output, index=False)

    print(f"\nSaved training dataset to:\n  {args.output.resolve()}")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
