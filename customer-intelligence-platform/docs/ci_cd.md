# CI/CD Notes

Personal notes and interpretation of this repo's CI/CD setup.

The actual workflow lives at the repo root, not in this project folder:
`data-science-portfolio/.github/workflows/ci.yml` — it's shared across
every project in the monorepo (`attribution/`, `customer-intelligence-platform/`,
`experimentation/`, `marketing-analytics/`, etc.), not scoped to this one.

# Production ML Architecture: From Model to Scalable, Privacy-Preserving System

When I think about production machine learning, I don't start with the algorithm. I think about the **end-to-end system**: data → features → training → validation → deployment → monitoring, with **privacy, scalability, reproducibility, and reliability built into every layer.**

## 1. Move From Notebooks to Modular Production Code

Notebooks are great for exploration and experimentation, but I wouldn't treat them as production systems.

I would move the model into a modular architecture:

```text
src/
├── data/
├── features/
├── preprocessing/
├── models/
├── evaluation/
├── privacy/
└── monitoring/

configs/
├── model_config.yaml
└── data_config.yaml

tests/
pipelines/
outputs/
```

The goal is **separation of concerns**.

Feature engineering, preprocessing, training, evaluation, privacy checks, and monitoring should be independent, reusable, testable modules.

Configuration should also be separated from business logic. YAML files can define things like:

* Data locations
* Feature definitions
* Hyperparameter ranges
* Training windows
* Evaluation thresholds
* Privacy or governance rules

That means I can change how a model is configured without rewriting the underlying production code.

---

# 2. Build Privacy Into the Architecture

For large-scale consumer ML, I would treat privacy as a **design constraint rather than a compliance check at the end**.

I would start with **data minimization**:

> What is the minimum amount of data the model actually needs to solve the problem?

That means avoiding unnecessary PII, using pseudonymous identifiers where appropriate, limiting feature retention, enforcing access controls, and ensuring sensitive attributes don't accidentally enter training datasets.

I would also think about whether learning needs to happen centrally at all.

Depending on the use case, architectures could include:

**On-device inference**

Keep computation close to the user whenever the device has sufficient compute.

**Federated learning**

Instead of centralizing raw user data, models can learn from distributed devices and aggregate model updates.

**Differential privacy**

Introduce mathematically controlled noise into aggregated information so useful population-level patterns can be learned while reducing the ability to infer information about an individual.

**Secure aggregation**

The server can aggregate model updates without needing visibility into an individual device's contribution.

The architectural principle is:

> **Move computation toward the data whenever possible rather than automatically moving data toward computation.**

---

# 3. PySpark Provides the Distributed Data Layer

Once you're operating across millions of customers, events, transactions, impressions, or device interactions, pandas-based pipelines can become a bottleneck.

This is where **PySpark** becomes important.

Spark partitions large datasets across a cluster and processes those partitions in parallel.

Instead of:

```text
1 machine
    ↓
500M rows
    ↓
Feature Engineering
```

I can distribute the workload:

```text
                Dataset
                   ↓
          Distributed Partitions
         ↙         ↓          ↘
      Worker 1   Worker 2   Worker N
         ↘         ↓          ↙
            Feature Dataset
```

That lets the same architecture scale across:

* Feature engineering
* Large joins
* Aggregations
* Training dataset construction
* Historical backfills
* Batch inference
* Model scoring

For example, instead of calculating 90-day customer behavior sequentially across hundreds of millions of events, Spark distributes that computation across many workers.

The important distinction is:

> **Spark doesn't necessarily make the model itself scalable—it makes the data and feature pipeline feeding the model scalable.**

For algorithms that support distributed training, the compute layer can also distribute model training.

---

# 4. Test the Pipeline, Not Just Model Accuracy

One of the biggest differences between experimental ML and production ML is testing.

I'm not only asking:

> "Does the model have good AUC?"

I'm asking:

> "Can I trust the system producing the model?"

For example, I would write tests ensuring that:

**Schema contracts haven't changed.**

If an upstream team changes:

```text
customer_order_date
```

to:

```text
cust_ord_dt
```

the pipeline should fail immediately rather than silently generating incorrect features.

**Categorical encoders handle unseen categories.**

If a new category appears in production that wasn't present during training, the inference pipeline should handle it predictably rather than failing or silently dropping records.

**Feature distributions remain reasonable.**

For example:

```text
historical average_order_value = $75

production average_order_value = $7,500
```

That should trigger investigation.

I would test for:

* Schema drift
* Missing values
* Unexpected categories
* Feature ranges
* Duplicate records
* Data leakage
* Training-serving skew
* Distribution drift

These tests create **data contracts between upstream data systems and ML systems.**

---

# 5. Full Model Reproducibility

Every production prediction should ultimately be traceable back to the system that created it.

Using an experiment-tracking/model-registry system such as MLflow, I would associate a model with:

```text
Model Version
      ↓
Git Commit
      ↓
YAML Configuration
      ↓
Feature Definitions
      ↓
Training Data Snapshot
      ↓
Hyperparameters
      ↓
Evaluation Metrics
```

That gives the organization lineage.

If Model V27 behaves differently from Model V26, I should be able to determine **exactly what changed**.

That becomes especially important when models affect millions of users.

---

# 6. CI/CD Creates the Automated Safety Net

If a data scientist pushes a new feature-engineering script, I don't want that code going directly into production.

The CI/CD pipeline should automatically:

```text
Git Push
   ↓
Build Container
   ↓
Run Unit Tests
   ↓
Validate Data Schema
   ↓
Run Privacy / Governance Checks
   ↓
Train Model
   ↓
Evaluate Model
   ↓
Compare Against Production Model
   ↓
Register Candidate
   ↓
Controlled Deployment
```

For example, deployment might require:

```text
AUC ≥ threshold
Calibration within threshold
No major feature drift
No schema failures
Latency ≤ threshold
Privacy checks passed
```

Only then does the candidate become eligible for deployment.

This gives teams the ability to **move quickly without sacrificing reliability.**

---

# 7. Scale Models Through Layered Architecture

At very large consumer scale, I wouldn't assume every request should hit a large centralized model.

I would think about a hierarchy:

```text
USER / DEVICE
      ↓
Can inference happen locally?
      ↓
     YES ─────→ On-device model

      NO
      ↓
Privacy-preserving compute
      ↓
Larger / more complex model
```

Models can also be optimized for the serving environment through techniques such as:

* Quantization
* Distillation
* Smaller specialized models
* Sparse architectures
* Hardware-aware optimization
* Distributed training
* Batch inference
* Caching
* Model routing

The objective isn't simply:

> "Build the most accurate model."

It is:

> **Find the best tradeoff between model quality, latency, compute cost, privacy, and user experience.**

---

# 8. Interpretability and Responsible ML

Performance alone isn't sufficient.

For tabular models, I would use techniques such as **SHAP** to understand:

* Global feature importance
* Individual prediction drivers
* Unexpected model behavior
* Potential proxy variables
* Model drift

But explainability is only one layer.

I would also evaluate:

```text
Accuracy
+ Calibration
+ Stability
+ Fairness
+ Privacy
+ Latency
+ Reliability
```

The question becomes not just:

> "Is the model accurate?"

but:

> **"Is this a model I would trust to operate reliably at scale?"**

---

# My Overall ML Architecture Philosophy

I think of production ML as five interconnected layers:

**1. Privacy**
Collect and expose only the data required for the problem.

**2. Scale**
Use distributed systems such as PySpark for large-scale data processing and feature engineering.

**3. Reliability**
Use schema validation, data-quality tests, unit tests, and monitoring.

**4. Reproducibility**
Track code, configuration, data, features, experiments, and model versions.

**5. Deployment Safety**
Use CI/CD, model registries, automated validation, controlled rollouts, and rollback mechanisms.

The model itself may only be one component.

**The real engineering challenge is building a trustworthy system around it.**





