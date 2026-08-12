"""
build_training_snapshots.py

Creates a leakage-safe, time-aware Customer 360 training dataset.

This script uses the feature-building functions in build_customer_360.py.
Place both Python files in the project root.

Run:

    python3 build_training_snapshots.py

Default output:

    data/processed/customer_training_snapshots.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from build_customer_360 import (
    build_customer_features,
    build_marketing_features,
    build_order_features,
    build_product_features,
    build_support_features,
    build_web_features,
    check_required_files,
    create_history_tables,
    handle_missing_values,
    load_data,
    validate_relationships,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe monthly Customer 360 training snapshots."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Folder containing raw CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/customer_training_snapshots.csv"),
        help="Output training dataset path.",
    )
    parser.add_argument(
        "--start-date",
        default="2025-01-01",
        help="First monthly snapshot date.",
    )
    parser.add_argument(
        "--end-date",
        default="2026-01-01",
        help="Last monthly snapshot date.",
    )
    return parser.parse_args()


def create_snapshot_dates(
    start_date: str,
    end_date: str,
) -> pd.DatetimeIndex:
    """Create one snapshot at the beginning of every month."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("--start-date must not be after --end-date.")

    return pd.date_range(start=start, end=end, freq="MS")


def merge_features_without_labels(
    customer_features: pd.DataFrame,
    order_features: pd.DataFrame,
    favorite_category: pd.DataFrame,
    category_count: pd.DataFrame,
    marketing_features: pd.DataFrame,
    web_features: pd.DataFrame,
    support_features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine customer-level features without using prebuilt labels."""
    return (
        customer_features
        .merge(
            order_features,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            favorite_category[
                [
                    "customer_id",
                    "favorite_category_by_spend",
                    "favorite_category_spend",
                ]
            ],
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            category_count,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            marketing_features,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            web_features,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            support_features,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
    )


def add_future_labels(
    features: pd.DataFrame,
    all_orders: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Create labels from activity AFTER the snapshot date.

    purchase_label_30d:
        1 when the customer places a completed order during the next 30 days.

    churn_label_180d:
        1 when an existing purchasing customer places no completed order
        during the next 180 days.

    future_revenue_180d:
        Sum of completed order revenue (order_total) during the next 180
        days. 0 for customers who place no completed order in that window.
    """
    completed_orders = all_orders.loc[
        all_orders["order_status"] == "Completed"
    ].copy()

    future_30d = completed_orders.loc[
        (completed_orders["order_date"] >= snapshot_date)
        & (
            completed_orders["order_date"]
            < snapshot_date + pd.Timedelta(days=30)
        )
    ]

    future_180d = completed_orders.loc[
        (completed_orders["order_date"] >= snapshot_date)
        & (
            completed_orders["order_date"]
            < snapshot_date + pd.Timedelta(days=180)
        )
    ]

    purchasers_30d = set(future_30d["customer_id"])
    purchasers_180d = set(future_180d["customer_id"])
    revenue_180d = future_180d.groupby("customer_id")["order_total"].sum()

    output = features.copy()
    output["snapshot_date"] = snapshot_date
    output["purchase_label_30d"] = (
        output["customer_id"].isin(purchasers_30d).astype(int)
    )
    output["churn_label_180d"] = (
        ~output["customer_id"].isin(purchasers_180d)
    ).astype(int)
    output["future_revenue_180d"] = (
        output["customer_id"].map(revenue_180d).fillna(0.0)
    )

    return output


def build_one_snapshot(
    tables: dict[str, pd.DataFrame],
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build one leakage-safe customer training snapshot."""
    history = create_history_tables(tables, snapshot_date)

    # Keep customers who existed at the snapshot date.
    eligible_customers = tables["customers"].loc[
        tables["customers"]["signup_date"] < snapshot_date
    ].copy()

    # Churn is defined only for customers with a completed purchase before
    # the snapshot date. Never-purchasers are not yet eligible to churn.
    historical_completed_orders = history["orders"].loc[
        history["orders"]["order_status"] == "Completed"
    ]
    existing_buyers = set(historical_completed_orders["customer_id"])
    eligible_customers = eligible_customers.loc[
        eligible_customers["customer_id"].isin(existing_buyers)
    ].copy()

    customer_features = build_customer_features(
        eligible_customers,
        snapshot_date,
    )

    order_features, completed_orders = build_order_features(
        history["orders"],
        snapshot_date,
    )

    favorite_category, category_count = build_product_features(
        tables["order_items"],
        completed_orders,
        tables["products"],
    )

    marketing_features = build_marketing_features(
        history["marketing"]
    )

    web_features = build_web_features(
        history["web_events"],
        snapshot_date,
    )

    support_features = build_support_features(
        history["support"]
    )

    snapshot = merge_features_without_labels(
        customer_features=customer_features,
        order_features=order_features,
        favorite_category=favorite_category,
        category_count=category_count,
        marketing_features=marketing_features,
        web_features=web_features,
        support_features=support_features,
    )

    snapshot = add_future_labels(
        snapshot,
        tables["orders"],
        snapshot_date,
    )

    # Use the existing business-aware missing-value logic.
    snapshot = handle_missing_values(snapshot)

    return snapshot


def validate_training_dataset(
    training_data: pd.DataFrame,
) -> None:
    """Validate the final customer-snapshot grain and labels."""
    print("\nValidating final training dataset")

    duplicate_rows = training_data.duplicated(
        subset=["customer_id", "snapshot_date"]
    ).sum()

    if duplicate_rows:
        raise ValueError(
            f"Found {duplicate_rows} duplicate customer-snapshot rows."
        )

    if training_data.isna().sum().sum() > 0:
        raise ValueError("Training dataset contains missing values.")

    for label in ["purchase_label_30d", "churn_label_180d"]:
        invalid = ~training_data[label].isin([0, 1])
        if invalid.any():
            raise ValueError(f"{label} contains values other than 0 and 1.")

    print(
        f"  Rows: {len(training_data):,}\n"
        f"  Columns: {training_data.shape[1]:,}\n"
        f"  Unique customers: "
        f"{training_data['customer_id'].nunique():,}\n"
        f"  Snapshot dates: "
        f"{training_data['snapshot_date'].nunique():,}"
    )

    print("\nLabel rates")
    print(
        training_data[
            ["purchase_label_30d", "churn_label_180d"]
        ].mean().round(4)
    )


def main() -> None:
    args = parse_arguments()

    print("=" * 72)
    print("TIME-AWARE CUSTOMER TRAINING SNAPSHOT PIPELINE")
    print("=" * 72)

    check_required_files(args.raw_dir)
    tables = load_data(args.raw_dir)
    validate_relationships(tables)

    snapshot_dates = create_snapshot_dates(
        args.start_date,
        args.end_date,
    )

    all_snapshots = []

    for snapshot_date in snapshot_dates:
        # Ensure the complete 180-day label window exists in the raw data.
        maximum_order_date = tables["orders"]["order_date"].max()
        required_end_date = snapshot_date + pd.Timedelta(days=180)

        if required_end_date > maximum_order_date + pd.Timedelta(days=1):
            print(
                f"\nSkipping {snapshot_date.date()}: "
                "insufficient future data for the 180-day label."
            )
            continue

        print("\n" + "-" * 72)
        print(f"BUILDING SNAPSHOT: {snapshot_date.date()}")
        print("-" * 72)

        snapshot = build_one_snapshot(
            tables,
            snapshot_date,
        )
        all_snapshots.append(snapshot)

        print(f"Snapshot rows: {len(snapshot):,}")

    if not all_snapshots:
        raise ValueError(
            "No snapshots were created. Check the requested dates and "
            "the available order-history range."
        )

    training_data = pd.concat(
        all_snapshots,
        ignore_index=True,
    )

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
