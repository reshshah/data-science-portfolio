"""
build_customer_360.py

Run this file from the root of your VS Code project:

    python3 build_customer_360.py

Expected structure:

customer intelligence platform/
├── build_customer_360.py
├── CUSTOMER_360_REFERENCE.md
└── data/
    ├── raw/
    │   ├── customers.csv
    │   ├── products.csv
    │   ├── orders.csv
    │   ├── order_items.csv
    │   ├── marketing_touches.csv
    │   ├── web_events.csv
    │   ├── support_tickets.csv
    │   └── customer_snapshot.csv
    └── processed/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "customers": "customers.csv",
    "products": "products.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "marketing": "marketing_touches.csv",
    "web_events": "web_events.csv",
    "support": "support_tickets.csv",
    "snapshot": "customer_snapshot.csv",
}

NEW_CUSTOMER_TENURE_DAYS = 90


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a validated Customer 360 model-ready feature table."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Folder containing raw CSV files. Default: data/raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/customer_model_features.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--cutoff-date",
        default="2026-07-01",
        help="Feature cutoff date in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def check_required_files(raw_dir: Path) -> None:
    missing = [
        filename
        for filename in REQUIRED_FILES.values()
        if not (raw_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {raw_dir.resolve()}: {missing}"
        )


def load_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    print("\nSTEP 1 — Loading CSV files")

    tables = {
        "customers": pd.read_csv(
            raw_dir / REQUIRED_FILES["customers"],
            parse_dates=["signup_date"],
        ),
        "products": pd.read_csv(
            raw_dir / REQUIRED_FILES["products"],
            parse_dates=["launch_date"],
        ),
        "orders": pd.read_csv(
            raw_dir / REQUIRED_FILES["orders"],
            parse_dates=["order_date"],
        ),
        "order_items": pd.read_csv(
            raw_dir / REQUIRED_FILES["order_items"]
        ),
        "marketing": pd.read_csv(
            raw_dir / REQUIRED_FILES["marketing"],
            parse_dates=["send_date"],
        ),
        "web_events": pd.read_csv(
            raw_dir / REQUIRED_FILES["web_events"],
            parse_dates=["event_timestamp"],
        ),
        "support": pd.read_csv(
            raw_dir / REQUIRED_FILES["support"],
            parse_dates=["created_date"],
        ),
        "snapshot": pd.read_csv(
            raw_dir / REQUIRED_FILES["snapshot"],
            parse_dates=["snapshot_date"],
        ),
    }

    for name, df in tables.items():
        print(f"  {name:15s}: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return tables


def validate_primary_key(
    df: pd.DataFrame,
    column: str,
    table_name: str,
) -> None:
    missing = int(df[column].isna().sum())
    duplicates = int(df[column].duplicated().sum())

    print(
        f"  {table_name}.{column}: "
        f"{missing} missing, {duplicates} duplicates"
    )

    if missing > 0:
        raise ValueError(
            f"{table_name}.{column} contains missing values."
        )

    if duplicates > 0:
        raise ValueError(
            f"{table_name}.{column} contains duplicate values."
        )


def validate_foreign_key(
    child_df: pd.DataFrame,
    child_column: str,
    parent_df: pd.DataFrame,
    parent_column: str,
    relationship_name: str,
) -> None:
    child_keys = set(child_df[child_column].dropna())
    parent_keys = set(parent_df[parent_column].dropna())
    orphan_keys = child_keys - parent_keys

    print(f"  {relationship_name}: {len(orphan_keys)} orphan keys")

    if orphan_keys:
        raise ValueError(
            f"{relationship_name} has invalid keys. "
            f"Examples: {list(orphan_keys)[:5]}"
        )


def validate_relationships(tables: dict[str, pd.DataFrame]) -> None:
    print("\nSTEP 2 — Validating primary keys")

    validate_primary_key(tables["customers"], "customer_id", "customers")
    validate_primary_key(tables["products"], "product_id", "products")
    validate_primary_key(tables["orders"], "order_id", "orders")
    validate_primary_key(
        tables["order_items"], "order_item_id", "order_items"
    )
    validate_primary_key(tables["marketing"], "touch_id", "marketing")
    validate_primary_key(tables["web_events"], "event_id", "web_events")
    validate_primary_key(tables["support"], "ticket_id", "support")

    print("\nSTEP 3 — Validating foreign keys")

    validate_foreign_key(
        tables["orders"], "customer_id",
        tables["customers"], "customer_id",
        "orders.customer_id -> customers.customer_id",
    )
    validate_foreign_key(
        tables["order_items"], "order_id",
        tables["orders"], "order_id",
        "order_items.order_id -> orders.order_id",
    )
    validate_foreign_key(
        tables["order_items"], "product_id",
        tables["products"], "product_id",
        "order_items.product_id -> products.product_id",
    )
    validate_foreign_key(
        tables["marketing"], "customer_id",
        tables["customers"], "customer_id",
        "marketing.customer_id -> customers.customer_id",
    )
    validate_foreign_key(
        tables["web_events"], "customer_id",
        tables["customers"], "customer_id",
        "web_events.customer_id -> customers.customer_id",
    )
    validate_foreign_key(
        tables["support"], "customer_id",
        tables["customers"], "customer_id",
        "support.customer_id -> customers.customer_id",
    )


def create_history_tables(
    tables: dict[str, pd.DataFrame],
    cutoff_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    print(f"\nSTEP 4 — Applying cutoff date: {cutoff_date.date()}")

    return {
        "orders": tables["orders"].loc[
            tables["orders"]["order_date"] < cutoff_date
        ].copy(),
        "marketing": tables["marketing"].loc[
            tables["marketing"]["send_date"] < cutoff_date
        ].copy(),
        "web_events": tables["web_events"].loc[
            tables["web_events"]["event_timestamp"] < cutoff_date
        ].copy(),
        "support": tables["support"].loc[
            tables["support"]["created_date"] < cutoff_date
        ].copy(),
    }


def build_customer_features(
    customers: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    print("\nSTEP 5 — Building base customer features")

    features = customers.copy()
    features["tenure_days"] = (
        cutoff_date - features["signup_date"]
    ).dt.days
    features["is_new_customer"] = (
        features["tenure_days"] <= NEW_CUSTOMER_TENURE_DAYS
    ).astype(int)

    for column in ["email_opt_in", "push_opt_in"]:
        features[column] = features[column].astype(int)

    return features


def build_order_features(
    orders_history: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\nSTEP 6 — Building RFM and order features")

    completed_orders = orders_history.loc[
        orders_history["order_status"] == "Completed"
    ].copy()

    order_features = (
        completed_orders.groupby("customer_id")
        .agg(
            lifetime_orders=("order_id", "nunique"),
            lifetime_revenue=("order_total", "sum"),
            avg_order_value=("order_total", "mean"),
            total_discount=("discount_amount", "sum"),
            avg_discount=("discount_amount", "mean"),
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
        )
        .reset_index()
    )

    order_features["days_since_last_purchase"] = (
        cutoff_date - order_features["last_order_date"]
    ).dt.days

    order_features["purchase_span_days"] = (
        order_features["last_order_date"]
        - order_features["first_order_date"]
    ).dt.days

    order_features["orders_per_active_month"] = (
        order_features["lifetime_orders"]
        / np.maximum(order_features["purchase_span_days"] / 30.0, 1)
    )

    print("\nSTEP 7 — Building rolling-window order features")

    for days in [30, 90, 180]:
        recent_orders = completed_orders.loc[
            completed_orders["order_date"]
            >= cutoff_date - pd.Timedelta(days=days)
        ]

        recent_features = (
            recent_orders.groupby("customer_id")
            .agg(
                **{
                    f"orders_{days}d": ("order_id", "nunique"),
                    f"revenue_{days}d": ("order_total", "sum"),
                }
            )
            .reset_index()
        )

        order_features = order_features.merge(
            recent_features,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )

    return order_features, completed_orders


def build_product_features(
    order_items: pd.DataFrame,
    completed_orders: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\nSTEP 8 — Building product and category features")

    item_detail = (
        order_items.merge(
            completed_orders[
                ["order_id", "customer_id", "order_date"]
            ],
            on="order_id",
            how="inner",
            validate="many_to_one",
        )
        .merge(
            products[
                ["product_id", "category", "brand", "margin_pct"]
            ],
            on="product_id",
            how="left",
            validate="many_to_one",
        )
    )

    category_spend = (
        item_detail.groupby(["customer_id", "category"])["line_total"]
        .sum()
        .reset_index()
    )

    favorite_category = (
        category_spend.sort_values(
            ["customer_id", "line_total"],
            ascending=[True, False],
        )
        .drop_duplicates("customer_id")
        .rename(
            columns={
                "category": "favorite_category_by_spend",
                "line_total": "favorite_category_spend",
            }
        )
    )

    category_count = (
        item_detail.groupby("customer_id")["category"]
        .nunique()
        .rename("categories_purchased")
        .reset_index()
    )

    return favorite_category, category_count


def build_marketing_features(
    marketing_history: pd.DataFrame,
) -> pd.DataFrame:
    print("\nSTEP 9 — Building marketing features")

    features = (
        marketing_history.groupby("customer_id")
        .agg(
            marketing_touches=("touch_id", "count"),
            email_opens=("opened", "sum"),
            marketing_clicks=("clicked", "sum"),
            marketing_conversions_7d=("converted_within_7d", "sum"),
            avg_offer_pct=("offer_pct", "mean"),
            marketing_cost=("cost", "sum"),
        )
        .reset_index()
    )

    features["open_rate"] = (
        features["email_opens"]
        / features["marketing_touches"].clip(lower=1)
    )
    features["click_rate"] = (
        features["marketing_clicks"]
        / features["marketing_touches"].clip(lower=1)
    )
    features["marketing_conversion_rate"] = (
        features["marketing_conversions_7d"]
        / features["marketing_touches"].clip(lower=1)
    )

    return features


def build_web_features(
    events_history: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    print("\nSTEP 10 — Building web journey features")

    session_summary = (
        events_history.groupby(["customer_id", "session_id"])
        .agg(
            session_start=("event_timestamp", "min"),
            session_end=("event_timestamp", "max"),
            events_in_session=("event_id", "count"),
            viewed_product=(
                "event_type",
                lambda x: int((x == "product_view").any()),
            ),
            viewed_reviews=(
                "event_type",
                lambda x: int((x == "review_view").any()),
            ),
            added_to_cart=(
                "event_type",
                lambda x: int((x == "add_to_cart").any()),
            ),
            started_checkout=(
                "event_type",
                lambda x: int((x == "checkout_start").any()),
            ),
            session_purchase=(
                "event_type",
                lambda x: int((x == "purchase").any()),
            ),
        )
        .reset_index()
    )

    session_summary["session_minutes"] = (
        session_summary["session_end"]
        - session_summary["session_start"]
    ).dt.total_seconds() / 60.0

    features = (
        session_summary.groupby("customer_id")
        .agg(
            sessions_lifetime=("session_id", "nunique"),
            avg_events_per_session=("events_in_session", "mean"),
            avg_session_minutes=("session_minutes", "mean"),
            product_view_sessions=("viewed_product", "sum"),
            review_view_sessions=("viewed_reviews", "sum"),
            cart_sessions=("added_to_cart", "sum"),
            checkout_sessions=("started_checkout", "sum"),
            web_purchase_sessions=("session_purchase", "sum"),
            last_session_date=("session_end", "max"),
        )
        .reset_index()
    )

    features["days_since_last_session"] = (
        cutoff_date - features["last_session_date"]
    ).dt.days

    features["session_conversion_rate"] = (
        features["web_purchase_sessions"]
        / features["sessions_lifetime"].clip(lower=1)
    )

    features["cart_to_purchase_rate"] = (
        features["web_purchase_sessions"]
        / features["cart_sessions"].clip(lower=1)
    )

    return features


def build_support_features(
    support_history: pd.DataFrame,
) -> pd.DataFrame:
    print("\nSTEP 11 — Building support features")

    return (
        support_history.groupby("customer_id")
        .agg(
            support_tickets=("ticket_id", "count"),
            avg_resolution_hours=("resolution_hours", "mean"),
            avg_csat=("csat_score", "mean"),
            unresolved_tickets=(
                "resolved",
                lambda x: int((~x.astype(bool)).sum()),
            ),
        )
        .reset_index()
    )


def merge_all_features(
    customer_features: pd.DataFrame,
    order_features: pd.DataFrame,
    favorite_category: pd.DataFrame,
    category_count: pd.DataFrame,
    marketing_features: pd.DataFrame,
    web_features: pd.DataFrame,
    support_features: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    print("\nSTEP 12 — Merging all feature families")

    feature_table = (
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

    print("\nSTEP 13 — Adding model labels")

    labels = snapshot[
        [
            "customer_id",
            "churn_label_180d",
            "purchase_propensity_label_30d",
        ]
    ]

    return feature_table.merge(
        labels,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )


def handle_missing_values(
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    print("\nSTEP 14 — Handling missing values")

    output = feature_table.copy()

    output["days_since_last_purchase"] = (
        output["days_since_last_purchase"]
        .fillna(output["tenure_days"])
    )

    output = output.drop(
        columns=[
            "signup_date",
            "first_order_date",
            "last_order_date",
            "last_session_date",
        ],
        errors="ignore",
    )

    numeric_columns = output.select_dtypes(include="number").columns
    output[numeric_columns] = output[numeric_columns].fillna(0)

    categorical_columns = output.select_dtypes(include="object").columns
    output[categorical_columns] = (
        output[categorical_columns].fillna("Unknown")
    )

    return output


def validate_final_table(
    feature_table: pd.DataFrame,
    customers: pd.DataFrame,
) -> None:
    print("\nSTEP 15 — Running final quality checks")

    if not feature_table["customer_id"].is_unique:
        raise ValueError("Final table has duplicate customer IDs.")

    if len(feature_table) != len(customers):
        raise ValueError(
            "Final row count does not match the customer table."
        )

    for label in [
        "churn_label_180d",
        "purchase_propensity_label_30d",
    ]:
        if feature_table[label].isna().any():
            raise ValueError(f"{label} contains missing values.")

    if feature_table.isna().sum().sum() > 0:
        raise ValueError("Final table still contains missing values.")

    print(
        f"  Final shape: "
        f"{feature_table.shape[0]:,} rows × "
        f"{feature_table.shape[1]:,} columns"
    )


def save_output(
    feature_table: pd.DataFrame,
    output_path: Path,
) -> None:
    print("\nSTEP 16 — Saving output")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(output_path, index=False)

    print(f"  Saved to: {output_path.resolve()}")


def main() -> None:
    args = parse_arguments()
    cutoff_date = pd.Timestamp(args.cutoff_date)

    print("=" * 70)
    print("CUSTOMER 360 PIPELINE")
    print("=" * 70)

    check_required_files(args.raw_dir)
    tables = load_data(args.raw_dir)
    validate_relationships(tables)

    history = create_history_tables(tables, cutoff_date)

    customer_features = build_customer_features(
        tables["customers"],
        cutoff_date,
    )

    order_features, completed_orders = build_order_features(
        history["orders"],
        cutoff_date,
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
        cutoff_date,
    )

    support_features = build_support_features(
        history["support"]
    )

    feature_table = merge_all_features(
        customer_features,
        order_features,
        favorite_category,
        category_count,
        marketing_features,
        web_features,
        support_features,
        tables["snapshot"],
    )

    feature_table = handle_missing_values(feature_table)
    validate_final_table(feature_table, tables["customers"])
    save_output(feature_table, args.output)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
