# Customer Intelligence Platform

An end-to-end ML platform for churn prediction and purchase propensity,
built on the real UCI "Online Retail II" transaction dataset.

## Project structure

```
customer intelligence platform/
├── data/
│   ├── raw/                        # online_retail_II.csv (gitignored — reproduce with pipelines/download_retail_data.py)
│   ├── processed/                  # retail_training_snapshots.csv
│   ├── ml/                         # churn target: train/validation/test parquet + feature_metadata.json
│   ├── ml_propensity/              # propensity target: same, for propensity_label_30d
│   └── ml/_splits/                 # shared split-assignment cache (computed once, reused by both targets)
├── pipelines/                      # data & feature engineering pipelines (run in order)
│   ├── download_retail_data.py     # fetches the UCI dataset -> data/raw/online_retail_II.csv
│   ├── build_retail_features.py    # transactions -> leakage-safe point-in-time snapshots (both labels)
│   ├── time_split.py               # shared, target-agnostic split logic (not run directly)
│   └── prepare_retail_ml_dataset.py # snapshots -> chronological train/validation/test parquet + metadata (--target selects the label)
├── configs/                        # one YAML per (task x model type): name, target, paths, hyperparameters
│   ├── churn_config.yaml               # churn, logistic regression
│   ├── churn_xgboost_config.yaml       # churn, XGBoost
│   ├── churn_lightgbm_config.yaml      # churn, LightGBM
│   ├── churn_routed_config.yaml        # churn, tenure-routed blend of the above two (see docs/MODEL_METRICS_NOTES.md)
│   ├── propensity_config.yaml          # propensity, logistic regression
│   ├── propensity_xgboost_config.yaml  # propensity, XGBoost
│   └── propensity_lightgbm_config.yaml # propensity, LightGBM
├── src/                            # shared library code used by models/ (data loading, preprocessing,
│                                    # feature validation, routing, training, evaluation, plotting, reporting)
├── models/                         # generic entrypoints, driven entirely by the config passed in
│   ├── train_classifier.py         # python3 models/train_classifier.py --config configs/<name>.yaml
│   └── predict_churn_routed.py     # evaluates the routed blend of two already-trained churn models
├── tests/                          # unit tests (pytest)
├── outputs/                        # generated model artifacts (gitignored, regenerate by rerunning models/)
│   ├── churn/{model_type}/         # models/  metrics/  predictions/  plots/  per model type, plus routed/
│   ├── propensity/{model_type}/
│   └── logs/
├── scripts/                        # ad hoc exploration scripts, not part of any pipeline
├── notebooks/                      # exploratory analysis (empty so far)
├── docs/                           # step-by-step reference guides for each pipeline stage + VS Code setup
├── customer-intelligence-env/      # local Python virtualenv (gitignored)
└── requirements.txt
```

## Setup

```bash
# create and activate the virtual environment (one-time)
python3 -m venv customer-intelligence-env
source customer-intelligence-env/bin/activate

# install dependencies
pip install -r requirements.txt
```

Every new session, just re-activate the env:

```bash
source customer-intelligence-env/bin/activate
```

In VS Code, select this environment via
**⌘⇧P → Python: Select Interpreter → `./customer-intelligence-env/bin/python`**.

More setup detail (including copy-paste terminal commands) is in `docs/vscode_setup.txt`.

## Running the pipeline end to end

```bash
cd "/Users/rshah/customer intelligence platform"
python3 pipelines/download_retail_data.py     # data/raw/online_retail_II.csv (~44MB, one-time)
python3 pipelines/build_retail_features.py    # data/processed/retail_training_snapshots.csv

# churn target
python3 pipelines/prepare_retail_ml_dataset.py --target churn_label_180d
python3 models/train_classifier.py --config configs/churn_config.yaml
python3 models/train_classifier.py --config configs/churn_xgboost_config.yaml
python3 models/train_classifier.py --config configs/churn_lightgbm_config.yaml
python3 models/predict_churn_routed.py --config configs/churn_routed_config.yaml

# propensity target (shares the same split as churn -- computed once, reused, not rebuilt)
python3 pipelines/prepare_retail_ml_dataset.py --target propensity_label_30d
python3 models/train_classifier.py --config configs/propensity_config.yaml
python3 models/train_classifier.py --config configs/propensity_xgboost_config.yaml
python3 models/train_classifier.py --config configs/propensity_lightgbm_config.yaml
```

`models/train_classifier.py` is fully generic — every run is driven by the `--config` passed
in (data location, target column, model type, hyperparameters), so adding a new model type or
a new target is a new YAML file, not new code.

Each pipeline stage has a step-by-step reference doc in `docs/`:
`RETAIL_FEATURES_REFERENCE.md`, `ML_DATASET_REFERENCE.md`. Start with
`docs/PLATFORM_ARCHITECTURE.md` for the full picture. Model evaluation
results are interpreted in `docs/MODEL_METRICS_NOTES.md`.

Run tests with:

```bash
python3 -m pytest -q
```

## Data

Real transaction data from the UCI "Online Retail II" dataset (Chen, D.
2012, CC BY 4.0, DOI 10.24432/C5CG6D) lives in `data/raw/`, fetched by
`pipelines/download_retail_data.py`.

| File | Rows | Description |
|---|---|---|
| `online_retail_II.csv` | 1,067,371 | UK online retailer's line-item transaction log, Dec 2009 – Dec 2011: `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country` |

`pipelines/build_retail_features.py` cleans this down to 824,293 usable
rows (5,939 customers) and builds leakage-safe, point-in-time customer
snapshots — 7 numeric features (recency, tenure, 90-day frequency/
monetary/basket-value/distinct-products/return-rate) plus two forward-
looking labels:

- **`churn_label_180d`** — no purchase in the 180 days after the snapshot
- **`propensity_label_30d`** — a purchase in the 30 days after the snapshot

16 monthly snapshots (2010-03 through 2011-06) yield **59,162 rows across
4,996 unique customers** in `data/processed/retail_training_snapshots.csv`.
Full detail in `docs/RETAIL_FEATURES_REFERENCE.md`.

## Tooling

`requirements.txt` is a broad exploration environment (Bayesian modeling,
forecasting, causal inference, deep learning, recommenders, etc.) for
future phases of the platform. What the current churn + propensity
pipeline actually uses: pandas, pyarrow (Parquet), scikit-learn, xgboost,
lightgbm, pyyaml, joblib, matplotlib, pytest.

## Roadmap

✅ **Phase 1 — Data Foundation**
- ✔ Retail transaction data (UCI Online Retail II) — replaced an earlier
  synthetic 200-customer dataset
- ✔ Leakage-safe point-in-time feature engineering
- ✔ Shared, cached train/validation/test split, reused across targets

▶ **Phase 2 (Now) — Model Development**
1. ✔ Train / Validation / Test Split
2. ✔ Baseline Model (logistic regression)
3. ✔ XGBoost / LightGBM Churn Models — on this dataset, logistic
   regression wins outright (test ROC-AUC 0.797 vs. 0.782 XGBoost / 0.776
   LightGBM) — see `docs/MODEL_METRICS_NOTES.md`
4. ✔ Model Evaluation (ROC-AUC, PR-AUC, precision/recall, threshold search)
5. ✔ Feature Importance
6. ✔ Purchase Propensity (`propensity_label_30d`) — same
   `models/train_classifier.py`, zero new code, only new config files.
   Test ROC-AUC 0.791 (logistic regression), PR-AUC 0.513 vs. a 0.158
   do-nothing floor
7. ✔ Cold-start routing investigated — re-checked on the retail data;
   logistic regression wins on both seen and genuinely-new customers here,
   so the tenure-routed ensemble (`churn_routed_config.yaml`) is currently
   *not* recommended over plain logistic regression (kept as a working
   example of the mechanism) — see `docs/MODEL_METRICS_NOTES.md`
8. ✔ Model Explainability (SHAP) — `mean(|SHAP value|)` importance +
   beeswarm plots for every trained model; `recency_days` dominates both
   churn and propensity by a wide margin — see `docs/MODEL_METRICS_NOTES.md`
9. ✔ Calibration curves, lift/gain chart, decile analysis — churn logistic
   regression is well-calibrated, not just well-ranked; top 3 deciles by
   risk capture 47.7% of actual churners
10. ✔ Hyperparameter tuning, cross-validation, MLflow run tracking —
    `models/tune_classifier.py` (RandomizedSearchCV + StratifiedKFold) and
    `src/tracking.py` (local MLflow store). Notable finding: the tuned
    LightGBM example scored *worse* on test (0.755 ROC-AUC) than the
    hand-set config (0.776) — tuning isn't free signal on this dataset,
    see `docs/MODEL_METRICS_NOTES.md`

**Not currently planned:** CLV/regression and uplift/causal modeling were
built against the earlier synthetic dataset and removed when the project
moved to real transaction data, which has neither a revenue label nor
treatment/control assignment data. Revisit if either becomes available.
See `docs/MODELING_GUIDE.md`.
