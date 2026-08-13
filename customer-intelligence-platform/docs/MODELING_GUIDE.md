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
| Classification (fixed hyperparameters) | `models/train_classifier.py` | `trainer.train_model` |
| Classification (CV hyperparameter search) | `models/tune_classifier.py` | `tuning.run_hyperparameter_search` |
| Routed churn scoring | `models/predict_churn_routed.py` | `routing.route_predict` |

Both training entry points run the same evaluation suite afterward
(diagnostics, MLflow logging — see below), so a tuned model and a
manually-configured one are directly comparable.

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

## Model Diagnostics (every training run)

`models/train_classifier.py` and `models/tune_classifier.py` both call
`src/diagnostics.run_diagnostics()` on the test set after fitting, in
addition to the existing ROC/PR curves and coefficient/split-gain feature
importance:

- **SHAP explainability** (`src/explain.py`) — `shap.TreeExplainer` for
  XGBoost/LightGBM, `shap.LinearExplainer` for logistic regression (both
  fast and exact for their model family; there's no need for the slow,
  approximate general-purpose explainer here). Saves a beeswarm summary
  plot (`plots/shap_summary.png`, magnitude *and* direction of each
  feature's effect, not just magnitude like split-gain/coefficients) and a
  `mean(|SHAP value|)` importance ranking (`metrics/shap_importance.csv`)
  that — unlike raw coefficients or split-gain — is on a comparable scale
  across model types.
- **Calibration** (`src/calibration.compute_calibration`) — does a
  predicted 0.7 actually mean "70% of the time this happens"? Quantile-
  binned predicted-vs-actual rate, saved as `metrics/calibration_table.csv`
  / `plots/calibration_curve.png`.
- **Lift / decile / gains** (`src/calibration.compute_decile_table`) — rank
  every test row into 10 equal-size buckets by predicted probability
  (decile 1 = highest). For each decile: actual positive rate, lift over
  the overall base rate, and cumulative capture. Answers the operational
  question directly: *"if we act on the top 20% by score, how many of the
  real churners/buyers do we actually catch?"* Saved as
  `metrics/decile_table.csv`, plus a lift chart and a cumulative gains
  chart (`plots/lift_chart.png`, `plots/gain_chart.png`).

## Hyperparameter Tuning

`models/tune_classifier.py` runs a cross-validated hyperparameter search
instead of fitting the fixed hyperparameters in `model:` directly. Add a
`tuning:` section to a config:

```yaml
model:                  # starting point -- overridden by whichever keys
  type: lightgbm         # tuning.param_distributions searches over
  n_estimators: 200
  ...

tuning:
  cv_folds: 5            # StratifiedKFold splits, within train only
  n_iter: 20             # random hyperparameter combinations to try
  scoring: roc_auc       # sklearn scoring string
  param_distributions:
    n_estimators: [100, 200, 300, 400]
    num_leaves: [7, 15, 31, 63]
    learning_rate: [0.01, 0.03, 0.05, 0.1]
```

See `configs/churn_lightgbm_tuned_config.yaml` for a full example.

```bash
python3 models/tune_classifier.py --config configs/churn_lightgbm_tuned_config.yaml
```

The search (`RandomizedSearchCV` + `StratifiedKFold`) never touches
validation or test — only train is cross-validated. The winning
hyperparameters are refit on the full training set, then evaluated on
validation/test exactly like `train_classifier.py` (same threshold search,
same diagnostics, same MLflow logging). Every trial's CV score is saved to
`metrics/cv_results.csv`; the winning hyperparameters to
`metrics/best_params.json`.

**Give a tuned config its own `name`** (e.g. `churn_tuned`, not `churn`) —
`output_dir`/`name`/`model.type` together form the output path, and a
tuned run sharing the untuned config's `name` will silently overwrite it.

**Worth knowing**: on this dataset, the example tuned LightGBM search
(`configs/churn_lightgbm_tuned_config.yaml`) actually landed on a *worse*
test ROC-AUC (0.755) than the hand-set LightGBM config (0.776), let alone
logistic regression (0.797) — see `MODEL_METRICS_NOTES.md`. Consistent
with this project's throughline (lower-capacity models generalize better
here because the dataset is small relative to tree-model capacity), a
wider search space that permits deeper/more complex trees found
CV-attractive-but-less-generalizable combinations. Tuning isn't free
signal; it optimizes exactly the objective and search space you give it.

## MLflow Run Tracking

Every `train_classifier.py`/`tune_classifier.py` run logs to a local
SQLite-backed MLflow store at `<project root>/mlruns.db` (metadata) +
`<project root>/mlruns/` (artifacts — model pickle, plots, metric CSVs),
via `src/tracking.py`. One experiment per task (`config["name"]`), one run
per invocation, named after the model type.

Logged per run: every `model:` hyperparameter, `target`, `random_state`,
dummy/validation/test metrics (scalars only — `confusion_matrix` isn't
loggable as a metric), the fitted pipeline, and every plot/metrics CSV as
artifacts. Tuning runs additionally log the search space, best CV score,
and winning hyperparameters.

Browse runs locally (no server needed):

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

`mlruns.db` and `mlruns/` are gitignored, same as `outputs/` — regenerate
by rerunning training.

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

# Hyperparameter tuning (CV search + full evaluation on the winning model)
python3 models/tune_classifier.py --config configs/churn_lightgbm_tuned_config.yaml
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
│   ├── feature_importance.csv           # coefficient / split-gain, per model type
│   ├── shap_importance.csv              # mean |SHAP value|, comparable across model types
│   ├── calibration_table.csv
│   ├── decile_table.csv                 # lift/gain source data
│   ├── cv_results.csv                   # tune_classifier.py runs only: every trial's CV score
│   └── best_params.json                 # tune_classifier.py runs only
├── predictions/
│   ├── validation_predictions.csv
│   └── test_predictions.csv
└── plots/                               # ROC/PR, feature importance, SHAP summary,
                                          # calibration curve, lift chart, gains chart
```

Logs go to `outputs/logs/`. All of `outputs/` is disposable and
gitignored — regenerate by rerunning the commands above. Same for
`mlruns.db`/`mlruns/` (MLflow's local store, see below).

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
