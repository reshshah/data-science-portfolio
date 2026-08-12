"""
build_uplift_dataset.py

Builds a leakage-safe treatment/control dataset for uplift modeling: for
each customer snapshot, attaches a treatment arm derived from marketing
touches in the 90 days before the snapshot date, and the existing
churn_label_180d as the outcome.

Marketing touches are randomized per touch, not per customer -- a customer
can receive both Treatment and Control touches over time. A snapshot is
only usable for uplift if its 90-day pre-window contains touches from
exactly one arm:
    - 0 touches in the window -> excluded (no experiment happened)
    - only Treatment touches -> arm = Treatment
    - only Control touches -> arm = Control
    - both arms present -> excluded (ambiguous attribution)

The marketing-engagement feature family (marketing_touches, email_opens,
marketing_clicks, marketing_conversions_7d, avg_offer_pct, marketing_cost,
open_rate, click_rate, marketing_conversion_rate) is excluded from the
covariate set: those features are computed from the same source table that
defines the treatment arm, so including them as CATE covariates risks
"bad controls" bias rather than genuine confounder adjustment.

Run from the project root:
    python3 pipelines/build_uplift_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_SNAPSHOTS = Path("data/processed/customer_training_snapshots.csv")
DEFAULT_TOUCHES = Path("data/raw/marketing_touches.csv")
DEFAULT_OUTPUT_DIR = Path("data/ml_uplift")

ATTRIBUTION_WINDOW_DAYS = 90
TREATMENT_COLUMN = "treatment"
OUTCOME_COLUMN = "churn_label_180d"
IDENTIFIER_COLUMNS = ["customer_id", "snapshot_date"]

# Computed from the same marketing_touches.csv rows that define the
# treatment arm -- excluded from covariates to avoid bad-controls bias.
MARKETING_ENGAGEMENT_FEATURES = [
    "marketing_touches",
    "email_opens",
    "marketing_clicks",
    "marketing_conversions_7d",
    "avg_offer_pct",
    "marketing_cost",
    "open_rate",
    "click_rate",
    "marketing_conversion_rate",
]

# Other targets present in customer_training_snapshots.csv, not used here.
OTHER_TARGET_COLUMNS = ["purchase_label_30d", "future_revenue_180d"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a treatment/control dataset for churn uplift modeling."
    )
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--touches", type=Path, default=DEFAULT_TOUCHES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-end", default="2025-09-01")
    parser.add_argument("--validation-end", default="2025-11-01")
    return parser.parse_args()


def assign_treatment_arms(
    snapshots: pd.DataFrame,
    touches: pd.DataFrame,
) -> pd.DataFrame:
    """Attach a clean Treatment/Control arm to each customer snapshot."""
    print("\nSTEP 1 — Assigning treatment arms from the 90-day pre-window")

    arm_frames = []
    for snapshot_date in sorted(snapshots["snapshot_date"].unique()):
        snapshot_date = pd.Timestamp(snapshot_date)
        window = touches.loc[
            (touches["send_date"] >= snapshot_date - pd.Timedelta(days=ATTRIBUTION_WINDOW_DAYS))
            & (touches["send_date"] < snapshot_date)
        ]
        arm = window.groupby("customer_id")["treatment_group"].agg(
            lambda groups: "mixed" if groups.nunique() > 1 else groups.iloc[0]
        )
        arm = arm[arm.isin(["Treatment", "Control"])]
        frame = arm.rename("arm").reset_index()
        frame["snapshot_date"] = snapshot_date
        arm_frames.append(frame)

    arms = pd.concat(arm_frames, ignore_index=True)

    merged = arms.merge(
        snapshots,
        on=["customer_id", "snapshot_date"],
        how="inner",
        validate="one_to_one",
    )
    merged[TREATMENT_COLUMN] = (merged["arm"] == "Treatment").astype(int)
    merged = merged.drop(columns=["arm"] + OTHER_TARGET_COLUMNS)

    dropped = len(arms) - len(merged)
    print(f"  Single-arm customer-touch-windows found: {len(arms):,}")
    if dropped:
        print(f"  Dropped ({dropped:,}) with no matching snapshot row (customer not yet active per snapshot criteria)")
    print(f"  Usable customer-snapshots: {len(merged):,}")
    print(f"  Treatment: {merged[TREATMENT_COLUMN].sum():,}")
    print(f"  Control:   {(merged[TREATMENT_COLUMN] == 0).sum():,}")

    return merged


def identify_feature_types(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    print("\nSTEP 2 — Detecting feature types")

    excluded = IDENTIFIER_COLUMNS + [TREATMENT_COLUMN, OUTCOME_COLUMN] + MARKETING_ENGAGEMENT_FEATURES
    feature_columns = [c for c in data.columns if c not in excluded]

    numeric_features = data[feature_columns].select_dtypes(include=["number"]).columns.tolist()
    categorical_features = data[feature_columns].select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Excluded (identifiers/targets/marketing-engagement): {len(excluded)}")

    return numeric_features, categorical_features


def create_time_splits(
    data: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\nSTEP 3 — Creating time-based splits")

    train = data.loc[data["snapshot_date"] <= train_end].copy()
    validation = data.loc[
        (data["snapshot_date"] > train_end) & (data["snapshot_date"] <= validation_end)
    ].copy()
    test = data.loc[data["snapshot_date"] > validation_end].copy()

    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        arm_counts = split[TREATMENT_COLUMN].value_counts().to_dict()
        print(
            f"  {name:10s}: {len(split):,} rows | "
            f"treatment={arm_counts.get(1, 0)} control={arm_counts.get(0, 0)} | "
            f"churn rate={split[OUTCOME_COLUMN].mean():.2%}"
        )

    return train, validation, test


def save_outputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    metadata: dict,
) -> None:
    print("\nSTEP 4 — Saving ML datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    validation.to_parquet(output_dir / "validation.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)

    with (output_dir / "feature_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    for name in ["train", "validation", "test", "feature_metadata"]:
        suffix = "parquet" if name != "feature_metadata" else "json"
        print(f"  {name:16s}: {(output_dir / f'{name}.{suffix}').resolve()}")


def main() -> None:
    args = parse_arguments()
    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)

    print("=" * 72)
    print("CHURN UPLIFT DATASET PREPARATION")
    print("=" * 72)

    snapshots = pd.read_csv(args.snapshots, parse_dates=["snapshot_date"])
    touches = pd.read_csv(args.touches, parse_dates=["send_date"])

    data = assign_treatment_arms(snapshots, touches)
    numeric_features, categorical_features = identify_feature_types(data)
    train, validation, test = create_time_splits(data, train_end, validation_end)

    metadata = {
        "treatment_column": TREATMENT_COLUMN,
        "outcome_column": OUTCOME_COLUMN,
        "identifier_columns": IDENTIFIER_COLUMNS,
        "excluded_marketing_engagement_features": MARKETING_ENGAGEMENT_FEATURES,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "attribution_window_days": ATTRIBUTION_WINDOW_DAYS,
        "train_end": str(train_end.date()),
        "validation_end": str(validation_end.date()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }

    save_outputs(train, validation, test, args.output_dir, metadata)
    print("\nUplift dataset preparation completed successfully.")


if __name__ == "__main__":
    main()
