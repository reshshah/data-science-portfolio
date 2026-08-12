## What This Platform Does

An end-to-end machine learning platform that turns raw customer data (orders, web events, support tickets, marketing touches) into trained, evaluated models for three tasks:

| Task | Target | Type |
|---|---|---|
| **Churn** | `churn_label_180d` | Binary classification (incl. cold-start routed ensemble) |
| **Propensity** | `purchase_label_30d` | Binary classification |
| **Uplift** | Treatment effect on churn | Causal ML (X-learner) |

The dataset is **synthetic and demo-scale (200 customers)** by design. The architecture — config-driven training, point-in-time-correct snapshots, feature store, routed models — is the artifact; the data is a prop.

## System Data Flow

```
RAW DATA  (data/raw/ — 8 source tables)
   ↓
pipelines/build_customer_360.py
   ↓
CUSTOMER 360  (data/processed/customer_model_features.csv)
   ↓
pipelines/build_training_snapshots.py
   ↓
TRAINING SNAPSHOTS  (data/processed/customer_training_snapshots.csv)
   ↓
pipelines/prepare_ml_dataset.py        pipelines/build_uplift_dataset.py
   ↓                                      ↓
MODEL-READY DATA  (data/ml*, train/validation/test parquet + feature_metadata.json)
   ↓
configs/   (declares task, target, model type, hyperparameters, threshold search)
   ↓
models/    (train_classifier.py / train_regressor.py / train_uplift.py entry points)
   ↓
src/       (reusable engine: loading, validation, preprocessing, training, evaluation)
   ↓
MODEL TRAINING
   ↓
outputs/   (pickled models, metrics JSON, predictions, plots, logs)
```

Each arrow is a script you can run independently. Rebuilding downstream artifacts never requires touching upstream code — only rerunning the stage whose inputs changed.
