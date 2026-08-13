"""
time_split.py

Target-agnostic, leakage-safe time-based train/validation/test split
assignment, shared across every ML dataset built from customer-snapshot
data (one row per customer_id + snapshot_date).

Split assignment depends only on --train-end/--validation-end, never on a
modeling target, so it can be computed once and cached under a shared
--splits-dir: the first caller computes and saves it, later callers
pointed at the same --splits-dir reuse it instead of re-deriving the
boundaries independently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

IDENTIFIER_COLUMNS = ["customer_id", "snapshot_date"]


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

    assignment = identifiers.copy()
    assignment["split"] = "test"
    assignment.loc[
        identifiers["snapshot_date"] <= validation_end, "split"
    ] = "validation"
    assignment.loc[
        identifiers["snapshot_date"] <= train_end, "split"
    ] = "train"

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
    """Reuse the cached split assignment if it matches this run, else build it."""
    print("\nSTEP — Loading or building split assignment")

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
            "The split cache is stale relative to the input data; rerun "
            "with --force-resplit."
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
    print("\nSTEP — Reviewing customer overlap")

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
        raise ValueError("Training and validation dates overlap.")

    if validation["snapshot_date"].max() >= test["snapshot_date"].min():
        raise ValueError("Validation and test dates overlap.")

    print("\nSTEP — Validating chronological split order")
    print("  Train before validation: passed")
    print("  Validation before test: passed")


def print_split_summary(
    split_name: str,
    split: pd.DataFrame,
    target: str,
    binary: bool = True,
) -> None:
    """Print row, customer, date, and target statistics."""
    target_mean = split[target].mean()

    print(f"\n{split_name.upper()} SUMMARY")
    print(f"  Rows: {len(split):,}")
    print(f"  Unique customers: {split['customer_id'].nunique():,}")
    print(f"  Snapshot dates: {split['snapshot_date'].nunique():,}")
    if binary:
        print(f"  Positive target rate: {target_mean:.2%}")
    else:
        print(f"  Target mean: {target_mean:,.2f}")
