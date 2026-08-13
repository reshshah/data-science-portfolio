# Modeling Guide

How models are trained in this platform: the modeling paths, which
algorithms are supported, how configs control everything, how to run
training, and where results land.

Read `PLATFORM_ARCHITECTURE.md` first for how modeling fits into the
overall data flow.

---

## The Modeling Paths

```
Classification
├── churn        (churn_label_180d — will this customer churn in the next 180 days?)
└── propensity   (propensity_label_30d — will this customer buy in the next 30 days?)
```

Both are binary classification, trained through the same entry point:

| Path | Entry point | Engine (in `src/`) |
|---|---|---|
| Classification | `models/train_classifier.py` | `trainer.train_model` |
| Routed churn scoring | `models/predict_churn_routed.py` | `routing.route_predict` |

There is currently no regression or causal/uplift modeling path in this
project — see the note at the bottom.

---

## Supported Models

Defined in `src/trainer.py`. Adding a model family means adding one
`_build_*` function and one dispatch line.

**Classification** (`model.type` in config):

| Type | Implementation | Imbalance handling |
|---|---|---|
| `logistic_regression` | sklearn `LogisticRegression` | `class_weight: balanced` |
| `xgboost` | `XGBClassifier` | `balance_classes: true` → `scale_pos_weight = n_neg / n_pos` |
| `lightgbm` | `LGBMClassifier` | `class_weight: balanced` |

**Baselines are mandatory.** Every classifier run first trains a
`DummyClassifier(strategy="prior")`. Baseline metrics are saved alongside
model metrics — a model that can't beat the dummy doesn't ship.

Every model is wrapped in a sklearn `Pipeline` with the shared preprocessor
(`src/preprocessing.build_preprocessor`), so preprocessing is fit on train
only and travels inside the pickle — no train/serve skew. (All 7 retail
features are numeric, so in practice this is just median-impute + scale;
the preprocessor's categorical branch exists for schema-agnosticism, not
because this dataset uses it.)

---

## How Configs Control Training

One YAML per (task, model family) in `configs/`. Code never changes to try
a new variant. Anatomy:

```yaml
name: churn                    # task name — becomes the outputs/ subfolder
target: churn_label_180d       # label column in the ML dataset
random_state: 42               # seeded everywhere for reproducibility

paths:
  data_dir: data/ml            # where train/validation/test parquet live
  metadata_file: data/ml/feature_metadata.json   # declares numeric/categorical features
  output_dir: outputs

model:
  type: lightgbm                # selects the _build_* function in src/trainer.py
  n_estimators: 200             # everything below is passed to that builder
  num_leaves: 15
  learning_rate: 0.05
  class_weight: balanced

threshold:                      # decision threshold search
  metric: f1                    # swept on the validation set
  search_min: 0.05
  search_max: 0.96
  search_step: 0.01

logging:
  level: INFO
  file: outputs/logs/train_churn_lightgbm.log
```

Key behaviors driven by config:

- **Feature selection** comes from `feature_metadata.json`, not the
  config — the dataset declares its own schema; training validates
  against it (`src/feature_validation.py`).
- **Threshold selection**: classifiers don't use 0.5. The threshold that
  maximizes the configured metric (F1) is found on the *validation* set,
  then applied unchanged to test. No test-set peeking.
- **Class imbalance** is handled per model family (see table above),
  toggled in the config.

### The routed churn ensemble

`configs/churn_routed_config.yaml` defines a two-model router
(`models/predict_churn_routed.py`): customers with `tenure_days <= 90` are
scored by the logistic-regression model; everyone else by LightGBM.
**On the current retail dataset this does not beat the plain logistic
regression model** — see `MODEL_METRICS_NOTES.md` for the numbers and why.
It's kept as a working example of the routing mechanism, not as the
recommended churn model right now.

---

## Running Training

From the project root, with the virtualenv active:

```bash
# Churn
python3 models/train_classifier.py --config configs/churn_config.yaml
python3 models/train_classifier.py --config configs/churn_lightgbm_config.yaml
python3 models/train_classifier.py --config configs/churn_xgboost_config.yaml

# Propensity
python3 models/train_classifier.py --config configs/propensity_config.yaml
python3 models/train_classifier.py --config configs/propensity_lightgbm_config.yaml
python3 models/train_classifier.py --config configs/propensity_xgboost_config.yaml

# Routed churn scoring (requires the logistic_regression and lightgbm churn models above)
python3 models/predict_churn_routed.py --config configs/churn_routed_config.yaml
```

To compare model families for a task, run each config variant and compare
`outputs/<task>/*/metrics/validation_metrics.json`. Select on validation;
report test once.

---

## Where Outputs Appear

Everything lands under `outputs/<name>/<model_type>/`:

```
outputs/churn/lightgbm/
├── models/lightgbm.pkl                  # full Pipeline (preprocessor + model)
├── metrics/
│   ├── dummy_validation.json            # baseline to beat
│   ├── validation_metrics.json          # includes selected threshold
│   ├── test_metrics.json
│   └── feature_importance.csv
├── predictions/
│   ├── validation_predictions.csv
│   └── test_predictions.csv
└── plots/                               # ROC/PR curves, feature importance
```

Logs go to `outputs/logs/`. All of `outputs/` is disposable and
gitignored — regenerate by rerunning the commands above.

How to *read* these metrics is covered in `MODEL_METRICS_NOTES.md`.

---

## Not currently in this project

`src/` still contains general-purpose regression and routing machinery
that isn't dead code, but has no live target to point it at:

- **Regression** (`compute_regression_metrics` in `src/evaluator.py`, the
  regression branch of `src/trainer.py`) — there's no regression target
  in the current schema (churn and propensity are both binary). This
  existed for a CLV target in an earlier, now-removed synthetic dataset.
  `models/train_regressor.py` was removed along with it; if a revenue/CLV
  target comes back, the regression code in `src/` is still there to
  build a new entry point against.
- **Causal / uplift modeling** — skipped for this dataset. The UCI Online
  Retail II transaction log has no treatment/control or campaign
  assignment data to estimate an incremental effect from. `src/uplift.py`
  and `models/train_uplift.py` were removed; revisit only if real
  treatment data becomes available for these customers.
