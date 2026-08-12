"""
prepare_ml_dataset.py

Prepares time-based train, validation, and test datasets from the
Customer 360 training snapshots.

Split assignment (which customer-snapshot rows fall into train/validation/
test) depends only on --train-end/--validation-end, not on the modeling
target. It is computed once and cached under --splits-dir; subsequent runs
for other targets (e.g. churn, purchase, CLV) reuse the cached assignment
instead of re-deriving the time boundaries, so every target-specific
dataset is guaranteed to share the same split.

Expected input:
    data/processed/customer_training_snapshots.csv

Default outputs:
    data/ml/train.parquet
    data/ml/validation.parquet
    data/ml/test.parquet
    data/ml/feature_metadata.json
    data/ml/_splits/split_assignment.parquet   (shared cache, reused by
                                                 other --output-dir runs
                                                 via matching --splits-dir)

Run from the project root:
    python3 prepare_ml_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/processed/customer_training_snapshots.csv")
DEFAULT_OUTPUT_DIR = Path("data/ml")
DEFAULT_SPLITS_DIR = Path("data/ml/_splits")

IDENTIFIER_COLUMNS = ["customer_id", "snapshot_date"]
BINARY_TARGET_COLUMNS = ["churn_label_180d", "purchase_label_30d"]
REGRESSION_TARGET_COLUMNS = ["future_revenue_180d"]
TARGET_COLUMNS = BINARY_TARGET_COLUMNS + REGRESSION_TARGET_COLUMNS


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description="Prepare time-based ML train, validation, and test datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to customer_training_snapshots.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Parquet outputs and metadata.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=DEFAULT_SPLITS_DIR,
        help=(
            "Directory holding the shared split assignment cache. "
            "Point multiple runs (different --target/--output-dir) at the "
            "same --splits-dir to guarantee they share one split."
        ),
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Recompute the split assignment even if a cached one exists.",
    )
    parser.add_argument(
        "--target",
        choices=TARGET_COLUMNS,
        default="churn_label_180d",
        help="Primary modeling target.",
    )
    parser.add_argument(
        "--train-end",
        default="2025-09-01",
        help="Last snapshot date included in training.",
    )
    parser.add_argument(
        "--validation-end",
        default="2025-11-01",
        help="Last snapshot date included in validation.",
    )
    return parser.parse_args()


def load_training_snapshots(input_path: Path) -> pd.DataFrame:
    """Load the time-aware customer snapshot dataset."""
    print("\nSTEP 1 — Loading training snapshots")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path.resolve()}"
        )

    data = pd.read_csv(
        input_path,
        parse_dates=["snapshot_date"],
    )

    print(
        f"  Loaded {len(data):,} rows × "
        f"{data.shape[1]:,} columns"
    )
    print(
        f"  Snapshot range: "
        f"{data['snapshot_date'].min().date()} to "
        f"{data['snapshot_date'].max().date()}"
    )

    return data


def validate_source_data(data: pd.DataFrame) -> None:
    """Check source grain, targets, dates, and missing values."""
    print("\nSTEP 2 — Validating source data")

    required_columns = set(
        IDENTIFIER_COLUMNS + TARGET_COLUMNS
    )
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    duplicate_count = data.duplicated(
        subset=["customer_id", "snapshot_date"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate customer-snapshot rows."
        )

    if data["snapshot_date"].isna().any():
        raise ValueError("snapshot_date contains missing values.")

    for target in TARGET_COLUMNS:
        if data[target].isna().any():
            raise ValueError(f"{target} contains missing values.")

    for target in BINARY_TARGET_COLUMNS:
        invalid = ~data[target].isin([0, 1])
        if invalid.any():
            raise ValueError(
                f"{target} contains values other than 0 and 1."
            )

    print("  Customer-snapshot grain: passed")
    print("  Required targets: passed")
    print("  Binary-target validation: passed")


def remove_leakage_and_unused_columns(
    data: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove identifiers and labels that should not be model inputs.

    The selected target remains in the prepared datasets.
    Every other known target column is removed to prevent one future
    outcome from being used to predict another.
    """
    print("\nSTEP 3 — Removing leakage and identifier columns")

    alternate_targets = [
        column for column in TARGET_COLUMNS if column != target
    ]

    model_data = data.drop(
        columns=alternate_targets,
        errors="ignore",
    ).copy()

    excluded_from_features = IDENTIFIER_COLUMNS + [target]

    print(f"  Primary target: {target}")
    print(f"  Removed alternate targets: {', '.join(alternate_targets)}")
    print(
        "  Excluded from model features: "
        + ", ".join(excluded_from_features)
    )

    return model_data, excluded_from_features


def identify_feature_types(
    data: pd.DataFrame,
    excluded_columns: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Identify numeric, categorical, and boolean feature columns."""
    print("\nSTEP 4 — Detecting feature types")

    feature_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

    numeric_features = (
        data[feature_columns]
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_features = (
        data[feature_columns]
        .select_dtypes(include=["object", "category"])
        .columns
        .tolist()
    )

    boolean_features = (
        data[feature_columns]
        .select_dtypes(include=["bool"])
        .columns
        .tolist()
    )

    print(f"  Total features: {len(feature_columns)}")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Boolean features: {len(boolean_features)}")

    unclassified = (
        set(feature_columns)
        - set(numeric_features)
        - set(categorical_features)
        - set(boolean_features)
    )

    if unclassified:
        raise ValueError(
            "Unclassified feature types found: "
            f"{sorted(unclassified)}"
        )

    return (
        numeric_features,
        categorical_features,
        boolean_features,
    )


def compute_split_assignment(
    identifiers: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Assign each customer-snapshot row to train/validation/test by date.

    Depends only on IDENTIFIER_COLUMNS, never on a modeling target, so the
    same assignment is valid for every target derived from this source.
    """
    if train_end >= validation_end:
        raise ValueError(
            "--train-end must be earlier than --validation-end."
        )

    conditions = [
        identifiers["snapshot_date"] <= train_end,
        identifiers["snapshot_date"] <= validation_end,
    ]
    labels = ["train", "validation"]

    assignment = identifiers.copy()
    assignment["split"] = "test"
    assignment.loc[conditions[1], "split"] = "validation"
    assignment.loc[conditions[0], "split"] = "train"

    for name in ("train", "validation", "test"):
        split_dates = assignment.loc[
            assignment["split"] == name, "snapshot_date"
        ]
        if split_dates.empty:
            raise ValueError(
                f"The {name} split is empty. "
                "Adjust the split dates to match the available snapshots."
            )

    return assignment


def fingerprint_identifiers(identifiers: pd.DataFrame) -> str:
    """Order-independent fingerprint of the customer-snapshot key set."""
    sorted_keys = identifiers.sort_values(IDENTIFIER_COLUMNS)
    return str(int(pd.util.hash_pandas_object(sorted_keys, index=False).sum()))


def load_or_build_split_assignment(
    data: pd.DataFrame,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    splits_dir: Path,
    force_resplit: bool,
) -> pd.DataFrame:
    """
    Reuse the cached split assignment if it matches this run, else build it.

    Caching here is what lets churn/purchase/CLV datasets share one split:
    the first run computes and saves it, later runs pointed at the same
    --splits-dir load it back instead of re-deriving train/validation/test
    boundaries independently.
    """
    print("\nSTEP 5 — Loading or building split assignment")

    assignment_path = splits_dir / "split_assignment.parquet"
    metadata_path = splits_dir / "split_assignment_metadata.json"
    identifiers = data[IDENTIFIER_COLUMNS]
    current_fingerprint = fingerprint_identifiers(identifiers)

    if not force_resplit and assignment_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            cached_metadata = json.load(file)

        cached_train_end = cached_metadata.get("train_end")
        cached_validation_end = cached_metadata.get("validation_end")
        cached_fingerprint = cached_metadata.get("identifier_fingerprint")

        if (
            cached_train_end == str(train_end.date())
            and cached_validation_end == str(validation_end.date())
            and cached_fingerprint == current_fingerprint
        ):
            assignment = pd.read_parquet(assignment_path)
            print(f"  Reusing cached split assignment: {assignment_path.resolve()}")
            print(
                "  (train_end, validation_end, and customer-snapshot keys "
                "match the cache — no resplit needed)"
            )
            return assignment

        print(
            "  Cached split assignment does not match this run "
            "(different --train-end/--validation-end or source data) — "
            "recomputing."
        )

    assignment = compute_split_assignment(
        identifiers,
        train_end,
        validation_end,
    )

    splits_dir.mkdir(parents=True, exist_ok=True)
    assignment.to_parquet(assignment_path, index=False)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "train_end": str(train_end.date()),
                "validation_end": str(validation_end.date()),
                "identifier_fingerprint": current_fingerprint,
                "row_count": len(assignment),
            },
            file,
            indent=2,
        )

    print(f"  Computed and cached new split assignment: {assignment_path.resolve()}")

    for name in ("train", "validation", "test"):
        split = assignment.loc[assignment["split"] == name]
        print(
            f"  {name:10s}: {len(split):,} rows | "
            f"{split['snapshot_date'].min().date()} to "
            f"{split['snapshot_date'].max().date()}"
        )

    return assignment


def apply_split_assignment(
    data: pd.DataFrame,
    assignment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slice a target-specific dataset using the shared split assignment."""
    merged = data.merge(
        assignment,
        on=IDENTIFIER_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    unmatched = merged["split"].isna().sum()
    if unmatched:
        raise ValueError(
            f"{unmatched} rows have no split assignment. "
            "The split cache is stale relative to --input; rerun with "
            "--force-resplit."
        )

    splits = {
        name: merged.loc[merged["split"] == name].drop(columns=["split"]).copy()
        for name in ("train", "validation", "test")
    }
    return splits["train"], splits["validation"], splits["test"]


def check_customer_overlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Report customer overlap.

    Overlap is expected in longitudinal customer data. The critical safeguard
    is that later snapshots never appear in earlier splits.
    """
    print("\nSTEP 6 — Reviewing customer overlap")

    train_customers = set(train["customer_id"])
    validation_customers = set(validation["customer_id"])
    test_customers = set(test["customer_id"])

    print(
        "  Train/validation shared customers: "
        f"{len(train_customers & validation_customers):,}"
    )
    print(
        "  Train/test shared customers: "
        f"{len(train_customers & test_customers):,}"
    )
    print(
        "  Validation/test shared customers: "
        f"{len(validation_customers & test_customers):,}"
    )
    print(
        "  Note: customer overlap is allowed because split assignment "
        "is based strictly on snapshot time."
    )


def validate_split_order(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Confirm chronological separation between splits."""
    if train["snapshot_date"].max() >= validation["snapshot_date"].min():
        raise ValueError(
            "Training and validation dates overlap."
        )

    if validation["snapshot_date"].max() >= test["snapshot_date"].min():
        raise ValueError(
            "Validation and test dates overlap."
        )

    print("\nSTEP 7 — Validating chronological split order")
    print("  Train before validation: passed")
    print("  Validation before test: passed")


def print_split_summary(
    split_name: str,
    split: pd.DataFrame,
    target: str,
) -> None:
    """Print row, customer, date, and target statistics."""
    target_mean = split[target].mean()

    print(f"\n{split_name.upper()} SUMMARY")
    print(f"  Rows: {len(split):,}")
    print(
        f"  Unique customers: "
        f"{split['customer_id'].nunique():,}"
    )
    print(
        f"  Snapshot dates: "
        f"{split['snapshot_date'].nunique():,}"
    )
    if target in BINARY_TARGET_COLUMNS:
        print(f"  Positive target rate: {target_mean:.2%}")
    else:
        print(f"  Target mean: {target_mean:,.2f}")


def save_outputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    metadata: dict,
) -> None:
    """Write split datasets and feature metadata."""
    print("\nSTEP 8 — Saving ML datasets")

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": output_dir / "train.parquet",
        "validation": output_dir / "validation.parquet",
        "test": output_dir / "test.parquet",
        "metadata": output_dir / "feature_metadata.json",
    }

    train.to_parquet(paths["train"], index=False)
    validation.to_parquet(paths["validation"], index=False)
    test.to_parquet(paths["test"], index=False)

    with paths["metadata"].open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    for name, path in paths.items():
        print(f"  {name:10s}: {path.resolve()}")


def main() -> None:
    args = parse_arguments()

    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)

    print("=" * 72)
    print("CUSTOMER INTELLIGENCE ML DATA PREPARATION")
    print("=" * 72)

    data = load_training_snapshots(args.input)
    validate_source_data(data)

    split_assignment = load_or_build_split_assignment(
        data,
        train_end,
        validation_end,
        args.splits_dir,
        args.force_resplit,
    )

    model_data, excluded_columns = (
        remove_leakage_and_unused_columns(
            data,
            args.target,
        )
    )

    (
        numeric_features,
        categorical_features,
        boolean_features,
    ) = identify_feature_types(
        model_data,
        excluded_columns,
    )

    train, validation, test = apply_split_assignment(
        model_data,
        split_assignment,
    )

    check_customer_overlap(train, validation, test)
    validate_split_order(train, validation, test)

    print_split_summary("train", train, args.target)
    print_split_summary("validation", validation, args.target)
    print_split_summary("test", test, args.target)

    metadata = {
        "target": args.target,
        "target_type": "binary" if args.target in BINARY_TARGET_COLUMNS else "regression",
        "identifier_columns": IDENTIFIER_COLUMNS,
        "excluded_from_features": excluded_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "boolean_features": boolean_features,
        "splits_dir": str(args.splits_dir.resolve()),
        "train_end": str(train_end.date()),
        "validation_end": str(validation_end.date()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_positive_rate": float(train[args.target].mean()),
        "validation_positive_rate": float(
            validation[args.target].mean()
        ),
        "test_positive_rate": float(test[args.target].mean()),
    }

    save_outputs(
        train,
        validation,
        test,
        args.output_dir,
        metadata,
    )

    print("\nML dataset preparation completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
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
