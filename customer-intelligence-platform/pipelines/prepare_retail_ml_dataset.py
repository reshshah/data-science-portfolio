"""
prepare_retail_ml_dataset.py

Prepares time-based train, validation, and test datasets for a chosen
retail target (churn or propensity) from the Online Retail II training
snapshots.

The two targets (churn_label_180d, propensity_label_30d) are built from the
same snapshots and share the same time-based split -- reused via
time_split.py's cache (compute once, reuse) -- so a churn model and a
propensity model trained from this script are directly comparable: same
rows, same split boundaries, different label.

Whichever target is NOT selected is dropped entirely from the saved
dataset: both labels are derived from the same future transactions, so
using one as a feature to predict the other would be label leakage.

Expected input:
    data/processed/retail_training_snapshots.csv

Default outputs (per --target):
    churn_label_180d      -> data/ml/{train,validation,test}.parquet
    propensity_label_30d  -> data/ml_propensity/{train,validation,test}.parquet
    data/ml/_splits/split_assignment.parquet  (shared cache, reused by both)

Run from the project root:
    python3 pipelines/prepare_retail_ml_dataset.py --target churn_label_180d
    python3 pipelines/prepare_retail_ml_dataset.py --target propensity_label_30d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import time_split
from build_retail_features import PROPENSITY_TARGET, RETAIL_FEATURES, TARGET, TARGET_COLUMNS

DEFAULT_INPUT = Path("data/processed/retail_training_snapshots.csv")
DEFAULT_SPLITS_DIR = Path("data/ml/_splits")
DEFAULT_OUTPUT_DIRS = {
    TARGET: Path("data/ml"),
    PROPENSITY_TARGET: Path("data/ml_propensity"),
}

IDENTIFIER_COLUMNS = ["customer_id", "snapshot_date"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare time-based ML train, validation, and test "
        "datasets for a retail target (churn or propensity)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--target",
        choices=TARGET_COLUMNS,
        default=TARGET,
        help="Which label to prepare a dataset for.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to data/ml for churn, data/ml_propensity for propensity.",
    )
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--force-resplit", action="store_true")
    parser.add_argument(
        "--train-end",
        default="2011-01-01",
        help="Last snapshot date included in training.",
    )
    parser.add_argument(
        "--validation-end",
        default="2011-04-01",
        help="Last snapshot date included in validation.",
    )
    return parser.parse_args()


def load_retail_snapshots(input_path: Path) -> pd.DataFrame:
    """Load the retail training snapshots built by build_retail_features.py."""
    print("\nSTEP 1 — Loading retail training snapshots")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    data = pd.read_csv(input_path, parse_dates=["snapshot_date"])

    print(f"  Loaded {len(data):,} rows x {data.shape[1]:,} columns")
    print(
        f"  Snapshot range: {data['snapshot_date'].min().date()} to "
        f"{data['snapshot_date'].max().date()}"
    )
    return data


def validate_source_data(data: pd.DataFrame) -> None:
    """Check source grain and both labels, regardless of which is selected."""
    print("\nSTEP 2 — Validating source data")

    required_columns = set(IDENTIFIER_COLUMNS + RETAIL_FEATURES + TARGET_COLUMNS)
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    duplicate_count = data.duplicated(subset=IDENTIFIER_COLUMNS).sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate customer-snapshot rows.")

    if data[RETAIL_FEATURES + TARGET_COLUMNS].isna().sum().sum() > 0:
        raise ValueError("Feature or target columns contain missing values.")

    for target in TARGET_COLUMNS:
        invalid = ~data[target].isin([0, 1])
        if invalid.any():
            raise ValueError(f"{target} contains values other than 0 and 1.")

    print("  Customer-snapshot grain: passed")
    print("  Required columns: passed")
    print("  Binary-target validation: passed")


def remove_alternate_targets(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Drop every label other than the selected one.

    churn_label_180d and propensity_label_30d are both derived from the
    same future transactions -- keeping the unselected one around risks it
    being used (accidentally or otherwise) as a feature, which is label
    leakage, not a legitimate predictor.
    """
    print("\nSTEP 3 — Removing alternate target columns")

    alternate_targets = [column for column in TARGET_COLUMNS if column != target]
    print(f"  Primary target: {target}")
    print(f"  Removed alternate targets: {', '.join(alternate_targets)}")

    return data.drop(columns=alternate_targets)


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
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS[args.target]

    print("=" * 72)
    print(f"RETAIL ML DATA PREPARATION — target: {args.target}")
    print("=" * 72)

    data = load_retail_snapshots(args.input)
    validate_source_data(data)

    # Split assignment depends only on identifiers/dates, computed from the
    # full (both-target) data so it's identical however this script is
    # invoked.
    split_assignment = time_split.load_or_build_split_assignment(
        data,
        train_end,
        validation_end,
        args.splits_dir,
        args.force_resplit,
    )

    model_data = remove_alternate_targets(data, args.target)

    train, validation, test = time_split.apply_split_assignment(model_data, split_assignment)

    time_split.check_customer_overlap(train, validation, test)
    time_split.validate_split_order(train, validation, test)

    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        time_split.print_split_summary(name, split, args.target)

    metadata = {
        "target": args.target,
        "target_type": "binary",
        "identifier_columns": IDENTIFIER_COLUMNS,
        "numeric_features": RETAIL_FEATURES,
        "categorical_features": [],
        "boolean_features": [],
        "splits_dir": str(args.splits_dir.resolve()),
        "train_end": str(train_end.date()),
        "validation_end": str(validation_end.date()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_positive_rate": float(train[args.target].mean()),
        "validation_positive_rate": float(validation[args.target].mean()),
        "test_positive_rate": float(test[args.target].mean()),
    }

    save_outputs(train, validation, test, output_dir, metadata)
    print("\nRetail ML dataset preparation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print(
            "\nERROR: Parquet support requires pyarrow.\n"
            "Install it with:\n"
            "  python3 -m pip install pyarrow",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
