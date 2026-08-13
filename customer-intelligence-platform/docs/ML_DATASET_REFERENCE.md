# Preparing Model-Ready Datasets

How `data/processed/retail_training_snapshots.csv` becomes chronological,
leakage-safe train/validation/test Parquet datasets — one per modeling
target, sharing one split.

Read `RETAIL_FEATURES_REFERENCE.md` first for how the snapshot table
itself is built.

---

## Why this step exists

The snapshot table has one row per `(customer_id, snapshot_date)`, with
both labels (`churn_label_180d`, `propensity_label_30d`) present. Two
things still need to happen before that's model-ready:

1. **Split chronologically**, not randomly — a customer's train-set row
   and test-set row can't straddle time in a way that leaks the future
   backward. Rows are assigned to train/validation/test purely by
   `snapshot_date`.
2. **Drop the label that isn't being modeled.** Both labels are derived
   from the same future transactions, so leaving `propensity_label_30d` in
   a churn model's feature set (or vice versa) would be label leakage, not
   a legitimate predictor.

## The split is computed once, shared by every target

`pipelines/time_split.py` holds the split logic, deliberately separate
from any target:

- `compute_split_assignment()` — assigns every `(customer_id,
  snapshot_date)` row to `train` / `validation` / `test` purely from
  `snapshot_date` vs. `--train-end` / `--validation-end`. Never looks at a
  label.
- `load_or_build_split_assignment()` — caches the result to
  `data/ml/_splits/split_assignment.parquet`, fingerprinted against the
  exact set of `(customer_id, snapshot_date)` keys plus the two cutoff
  dates. A second call with matching inputs reuses the cache instead of
  recomputing; a call with different inputs (different dates, or the
  source data changed) detects the mismatch and rebuilds.
- `apply_split_assignment()` — merges a target-specific dataframe against
  the cached assignment and slices it into train/validation/test.

This is why `data/ml/` (churn) and `data/ml_propensity/` (propensity) are
**not** two independent splits — building the second one prints "Reusing
cached split assignment" rather than re-deriving the boundaries. Same
rows, same `train_end`/`validation_end`, verified in both folders'
`feature_metadata.json`. See [[split_once_reuse_architecture]].

## Running it

```bash
python3 pipelines/prepare_retail_ml_dataset.py --target churn_label_180d
python3 pipelines/prepare_retail_ml_dataset.py --target propensity_label_30d
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `data/processed/retail_training_snapshots.csv` | Source snapshot table |
| `--target` | `churn_label_180d` | `churn_label_180d` or `propensity_label_30d` |
| `--output-dir` | `data/ml` for churn, `data/ml_propensity` for propensity | Where the Parquet + metadata land |
| `--splits-dir` | `data/ml/_splits` | Shared split-assignment cache — point every target at the same dir |
| `--force-resplit` | off | Recompute the split even if a matching cache exists |
| `--train-end` | `2011-01-01` | Last snapshot date in train |
| `--validation-end` | `2011-04-01` | Last snapshot date in validation |

## What happens, step by step

1. **Load** the snapshot CSV.
2. **Validate** — required columns present, no duplicate
   `(customer_id, snapshot_date)` rows, no missing values, both labels
   strictly 0/1 (checked regardless of which target is selected, so a
   schema problem in either label is caught early).
3. **Load or build the split assignment** (see above).
4. **Drop the alternate target** — whichever of `churn_label_180d` /
   `propensity_label_30d` wasn't selected is removed from the dataframe
   entirely, not just excluded from a features list. It never reaches the
   saved Parquet files.
5. **Apply the split**, check customer overlap (expected — the same
   customer can appear in multiple months, just never a later snapshot in
   an earlier split), and confirm chronological order
   (`train.max(snapshot_date) < validation.min(...) < test.min(...)`).
6. **Save** `train.parquet`, `validation.parquet`, `test.parquet`, and
   `feature_metadata.json` (target, feature lists, split boundaries, row
   counts, positive rates per split).

## Current split (both targets)

| Split | Snapshot months | Rows | Churn rate | Propensity rate |
|---|---|---|---|---|
| train | 2010-03 → 2011-01 (11 months) | 35,408 | 39.7% | 25.4% |
| validation | 2011-02 → 2011-04 (3 months) | 13,870 | 54.2% | 15.9% |
| test | 2011-05 → 2011-06 (2 months) | 9,884 | 50.6% | 18.3% |

Both `data/ml/feature_metadata.json` and `data/ml_propensity/feature_metadata.json`
list the same 7 numeric features (`RETAIL_FEATURES` — see
`RETAIL_FEATURES_REFERENCE.md`), no categorical or boolean features.

## Next step

`models/train_classifier.py --config configs/<name>.yaml` trains against
whichever `data_dir` the config points at — see `MODELING_GUIDE.md`.
