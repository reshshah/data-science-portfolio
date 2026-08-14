# Production ML Architecture: From Model to Scalable, Privacy-Preserving System
→ 1. Mental model: BUILD → SCALE → TEST → TRACK → DEPLOY → PROTECT
→ 2. BigQuery & scaling
→ 3. Modular Python architecture
→ 4. Testing & validation
→ 5. MLflow & reproducibility
→ 6. CI/CD
→ 7. Privacy
→ 8. What my repo currently implements
→ 9. What I would add in production

# Production ML Architecture & MLOps Notes

Personal reference for how I think about taking a machine learning model from development to a scalable, testable, reproducible, and privacy-aware production system.

## Core Framework

BUILD → SCALE → TEST → TRACK → DEPLOY → PROTECT

1. BUILD — BigQuery + modular Python architecture
2. SCALE — Large-scale processing and feature engineering in BigQuery
3. TEST — Unit, schema, data quality, and pipeline validation
4. TRACK — Git + YAML configuration + data snapshots + MLflow
5. DEPLOY — GitHub + CI/CD + automated validation
6. PROTECT — Data minimization, access controls, and privacy-aware architecture

---

## CI/CD Setup in This Repository

The actual CI/CD workflow lives at the repository root: `data-science-portfolio/.github/workflows/ci.yml`

The workflow is shared across projects in the monorepo, including:

- `attribution/`
- `customer-intelligence-platform/`
- `experimentation/`
- `marketing-analytics/`

It is therefore a portfolio-level CI/CD workflow rather than one scoped specifically to the Customer Intelligence Platform.

The purpose of CI/CD is to provide an automated safety layer between changes to the codebase and production deployment.

Conceptually:

Code Change
    ↓
GitHub Push / Pull Request
    ↓
CI/CD Pipeline
    ↓
Unit Tests
    ↓
Schema & Data Validation
    ↓
Model / Pipeline Validation
    ↓
Deployment Approval
    ↓
Production

I think about production machine learning as the **end-to-end system**: data → features → training → validation → deployment → monitoring, with **privacy, scalability, reproducibility, and reliability built into every layer.**

This can be explained across six areas: code, scale, testing, reproducibility, deployment, and privacy.I typically start with the data in BigQuery and use SQL to do the heavy data processing and feature creation. One advantage of BigQuery is that it already gives me a distributed, scalable compute layer, so if I'm working with hundreds of millions of customer transactions or events, I don't necessarily need to move all that data locally to process it.

Once I know the model works, I modularize the Python code so feature engineering, training, and evaluation are independently testable. I separate configuration from the code, put automated tests around the data and pipeline, and use something like MLflow so every model can be traced back to its code, configuration, training data, and evaluation metrics. Then CI/CD becomes the safety net before anything reaches production. And especially with consumer data, privacy needs to be part of the architecture from the beginning—minimize the data I'm using and avoid moving sensitive data unnecessarily.

Here is how I scale my model: Because I work with BigQuery, my first approach would be to push the heavy computation down to BigQuery rather than extracting massive datasets into Python. BigQuery gives me distributed compute for the joins, aggregations and feature engineering. I can then train the model in Python, or for appropriate use cases use BigQuery ML and keep training and inference closer to the data. That also has a privacy benefit because I'm minimizing unnecessary movement of customer-level data

**1. Is the code production-ready?**

BigQuery + SQL → Modular Python code → Configuration

I typically start with the data in BigQuery, where I use SQL for data exploration, joins, aggregations, and initial feature creation.

For the ML components, I structure the Python code into separate, testable modules:

Feature engineering → Preprocessing → Model training → Evaluation

I also separate configuration from the actual code. YAML files can store things like data paths, feature definitions, hyperparameter ranges, and model thresholds.

This makes the code easier to test, maintain, reuse, and scale across different models.

Example:

“I separate the data processing, model logic, and configuration so each component can be independently tested and changed without impacting the entire pipeline.”

**2. Can it handle scale?**

BigQuery → Large-scale data processing → Model-ready dataset

Because the data is already in BigQuery, I push the heavy computation down to BigQuery rather than pulling hundreds of millions of rows into Python.

BigQuery handles the distributed processing behind the scenes.

I use SQL for:

Large joins → Aggregations → Feature creation → Training dataset creation

For example, if I have hundreds of millions of customer transactions or behavioral events, I can use BigQuery to turn those into customer-level features such as:

90-day spend | purchase frequency | category engagement | site activity | marketing engagement

Python can then work with the resulting model-ready dataset rather than processing all the raw events.

For appropriate use cases, BigQuery ML can also train and score models directly where the data lives.

Example:

“I push the heavy computation to BigQuery. That gives me scalable distributed processing without having to move massive raw datasets into Python.”

**3. Can I trust the pipeline?**

Unit tests + Data tests + Schema tests

I'm not just testing whether the model is accurate. I'm testing whether I can trust the data and pipeline producing that model.

For example:

Did an upstream column name change?
Did a new categorical value appear?
Did null rates suddenly increase?
Did a feature distribution change dramatically?
Did something introduce data leakage?

These issues should be caught automatically before they affect the model.

Example:

“Before I trust the model output, I need to trust the data and pipeline producing it.”

**4. Can I reproduce the model?**

Git + YAML + BigQuery data snapshot + MLflow

For every production model, I want to know exactly:

Code + Configuration + Data + Features + Hyperparameters = Model version

Git tracks the code.

YAML tracks the configuration.

BigQuery provides the training dataset.

And something like MLflow can track experiments, metrics, artifacts, and model versions.

So if one model behaves differently from another, I can trace exactly what changed.

Example:

“If Model 27 behaves differently from Model 26, I should be able to trace that difference back to the code, configuration, data, or model parameters.”

**5. Can I deploy changes safely?**

GitHub → CI/CD → Tests → Model → Production

If I change feature-engineering or model code and push it to GitHub, I don't want that change automatically impacting production.

CI/CD provides the safety net:

Code change → Unit tests → Schema validation → Data-quality checks → Train → Evaluate → Compare → Deploy

If something fails, the pipeline stops.

Example:

“CI/CD lets the team iterate faster because we're automating the checks that protect production.”

**6. Is privacy built into the architecture?**

I would start with a very simple question:

“Do I actually need this customer data to solve the problem?”

That means minimizing the data used and avoiding unnecessary movement of sensitive customer-level data.

BigQuery also helps with this architecture because I can perform large joins, aggregations, and feature creation where the governed data already lives rather than unnecessarily extracting raw customer-level data into other environments.

For certain use cases, I would also think about:

On-device inference — Process information locally when possible.

Federated learning — Learn across devices without centrally collecting all the raw individual-level data.

Differential privacy — Learn population-level patterns while reducing the ability to identify an individual.

Example:

“Instead of automatically moving the data to the model, I think about whether I can move the computation closer to where the data already lives.”


