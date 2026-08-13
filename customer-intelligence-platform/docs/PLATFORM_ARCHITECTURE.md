# Platform Architecture

**Start here.** This is the orientation document for the Customer Intelligence Platform — the one to read first when returning to the project after time away. It explains how the system fits together, where every piece lives, and which document to read next for any given question.

---

## What This Platform Does

An end-to-end machine learning platform that turns a raw retail
transaction log into trained, evaluated models for two tasks:

| Task | Target | Type |
|---|---|---|
| **Churn** | `churn_label_180d` | Binary classification (incl. a tenure-routed ensemble variant) |
| **Propensity** | `propensity_label_30d` | Binary classification |

The data is the real UCI "Online Retail II" dataset (Chen, D. 2012, CC BY
4.0) — 1,067,371 line-item transactions from a UK online retailer,
Dec 2009–Dec 2011, 4,996 customers after cleaning. The architecture
— config-driven training, point-in-time-correct snapshots, a shared
leakage-safe split reused across targets — is the artifact worth
preserving even as the underlying dataset changes.

*Earlier phases of this project used a synthetic 200-customer dataset with
additional CLV (regression) and uplift (causal) targets. Both were removed
when the project moved to real transaction data, which has no revenue
label or treatment/control assignment. See `MODELING_GUIDE.md`'s "Not
currently in this project" section if you're looking for either.*

---

## System Data Flow

```
RAW DATA  (data/raw/online_retail_II.csv — UCI Online Retail II transaction log)
   ↓
pipelines/download_retail_data.py
   ↓
pipelines/build_retail_features.py
   ↓
RETAIL TRAINING SNAPSHOTS  (data/processed/retail_training_snapshots.csv —
                             both churn_label_180d and propensity_label_30d)
   ↓
pipelines/prepare_retail_ml_dataset.py  (--target selects the label;
                                          pipelines/time_split.py computes
                                          the split once, shared by both)
   ↓
MODEL-READY DATA  (data/ml/, data/ml_propensity/ — train/validation/test
                    parquet + feature_metadata.json)
   ↓
configs/   (declares task, target, model type, hyperparameters, threshold search)
   ↓
models/    (train_classifier.py, predict_churn_routed.py)
   ↓
src/       (reusable engine: loading, validation, preprocessing, training, evaluation)
   ↓
MODEL TRAINING
   ↓
outputs/   (pickled models, metrics JSON, predictions, plots, logs)
```

Each arrow is a script you can run independently. Rebuilding downstream
artifacts never requires touching upstream code — only rerunning the
stage whose inputs changed.

---

## Documentation Map

Read in this order the first time; jump directly to the relevant doc afterward.

```
PLATFORM_ARCHITECTURE  (this doc)
        ↓  How does the whole system work?
RETAIL_FEATURES_REFERENCE
        ↓  How do we build point-in-time retail features?
ML_DATASET_REFERENCE
        ↓  How do we create model-ready datasets, and share one split across targets?
MODELING_GUIDE
        ↓  How do we train models?
MODEL_METRICS_NOTES
        ↓  How do we evaluate them?
```

| Document | Answers |
|---|---|
| `docs/PLATFORM_ARCHITECTURE.md` | How does the whole system work? |
| `docs/RETAIL_FEATURES_REFERENCE.md` | How do we build point-in-time retail features? |
| `docs/ML_DATASET_REFERENCE.md` | How do we create model-ready datasets? |
| `docs/MODELING_GUIDE.md` | How do we train models? |
| `docs/MODEL_METRICS_NOTES.md` | How do we evaluate them? |

---

## Folder Guide

```
data/          Data
pipelines/     Build datasets
configs/       Model configuration
src/           Reusable ML engine
models/        Training/prediction entry points
outputs/       Model artifacts/results
tests/         Automated testing
scripts/       Ad hoc exploration
docs/          Documentation
```

### `data/` — Data

- `raw/online_retail_II.csv` — the UCI transaction log (reproduce with
  `pipelines/download_retail_data.py`; gitignored, not committed).
- `processed/retail_training_snapshots.csv` — leakage-safe point-in-time
  snapshots with both labels.
- `ml/` (churn), `ml_propensity/` (propensity) — model-ready
  train/validation/test Parquet splits, each with a `feature_metadata.json`
  declaring numeric features. `ml/_splits/` holds the shared split cache
  both folders are built from — see `ML_DATASET_REFERENCE.md`.

### `pipelines/` — Build datasets

One script per transformation stage. `download_retail_data.py` fetches
the raw data. `build_retail_features.py` builds point-in-time customer
snapshots with both labels. `time_split.py` holds target-agnostic split
logic (compute once, cache, reuse — not a script you run directly).
`prepare_retail_ml_dataset.py` produces the per-target train/validation/
test splits.

### `configs/` — Model configuration

One YAML per (task, model family) combination — e.g.
`churn_lightgbm_config.yaml`. Configs declare the target, data paths,
model type and hyperparameters, and threshold-search settings. Training
behavior changes by editing a config, never by editing code.
`churn_routed_config.yaml` defines a tenure-routed ensemble (currently
not recommended over plain logistic regression on this dataset — see
`MODEL_METRICS_NOTES.md`).

### `src/` — Reusable ML engine

Small, single-purpose, importable modules shared by every entry point:
`data_loader`, `feature_validation`, `preprocessing`, `trainer`,
`evaluator`, `routing`, `plots`, `reporting`, `utils`. No entry-point
script contains modeling logic — it all lives here. (`trainer.py` and
`evaluator.py` still carry generic regression support, unused by the
current classification-only targets — see `MODELING_GUIDE.md`.)

### `models/` — Training/prediction entry points

Thin CLI wrappers around `src/`: `train_classifier.py` (churn,
propensity), `predict_churn_routed.py` (tenure-routed scoring). Both take
`--config <path>`.

### `outputs/` — Model artifacts/results

Generated per task/model: `models/` (pickles), `metrics/` (validation +
test JSON, feature importance), `predictions/` (scored CSVs), `plots/`
(ROC/PR, importance), `logs/`. Gitignored — regenerate by rerunning
`models/` scripts. Never edited by hand.

### `tests/` — Automated testing

Pytest suite covering point-in-time feature construction (the
leakage-safety guarantees), preprocessing, regression-engine internals,
and routing logic. Run with `pytest` from the project root.

### `scripts/` — Ad hoc exploration

`read_parquet.py` — quick inspection of a prepared dataset, not part of
any pipeline.

### `docs/` — Documentation

You are here.

---

## Common Workflows

**Rebuild everything from raw data:**

```bash
python3 pipelines/download_retail_data.py
python3 pipelines/build_retail_features.py
python3 pipelines/prepare_retail_ml_dataset.py --target churn_label_180d
python3 pipelines/prepare_retail_ml_dataset.py --target propensity_label_30d
```

**Train a model:**

```bash
python3 models/train_classifier.py --config configs/churn_lightgbm_config.yaml
```

**Compare model families for a task:** run each config variant
(`churn_config.yaml`, `churn_lightgbm_config.yaml`,
`churn_xgboost_config.yaml`), then compare
`outputs/churn/*/metrics/validation_metrics.json`.

**Score with the routed churn ensemble:**

```bash
python3 models/predict_churn_routed.py --config configs/churn_routed_config.yaml
```

**Run tests:**

```bash
pytest
```

---

## Design Principles

1. **Config-driven, not code-driven.** New model variant = new YAML, zero code changes.
2. **Point-in-time correctness.** Training snapshots only use information available as of the snapshot date — no leakage from the future, and no label used as a feature for a different label derived from the same future window.
3. **Split once, reuse everywhere.** The train/validation/test boundary is computed from dates alone, cached, and shared across every target — never re-derived per model. See `ML_DATASET_REFERENCE.md`.
4. **Thin entry points, fat library.** Scripts orchestrate; `src/` implements. Everything in `src/` is importable and unit-testable.
5. **Artifacts are disposable.** Everything in `outputs/` and `data/processed/`/`data/ml*` can be regenerated from raw data plus code.
6. **Baselines always.** Every classifier run trains a dummy baseline first; a model that can't beat it doesn't ship.
