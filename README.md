# Applied ML & Analytics Portfolio

Self-contained projects built from real marketing, growth, and experimentation problems in retail. Each one exists to answer a specific business question a marketing or growth team actually faces — the sections below are organized in the same order as the folders in this repo.

---

## 🔀 Attribution
**`attribution/`**

Multi-touch attribution: given user-level touchpoint paths, which channel deserves credit for a conversion? Where marketing mix modeling answers budget questions top-down at the channel level, attribution works bottom-up from individual journeys — the two disagree by design, and teams need both.

*The business question:* "Which channels are actually driving conversions — and which are just getting credit?"

<!-- TODO: replace with actual file names and one-line descriptions once listed -->

### 🧭 Customer Intelligence Platform
**`customer-intelligence-platform/`**

An end-to-end ML platform that turns raw customer data (orders, web events, support tickets, marketing touches) into trained, evaluated models for three tasks: churn (180-day binary classification with cold-start routing), purchase propensity (30-day), and uplift (X-learner causal ML for treatment effects). Config-driven training, point-in-time-correct snapshots, and a feature store — the architecture is the artifact; the demo-scale synthetic data (200 customers) is a prop.

*The business questions:* "Which customers are about to leave, which are ready to buy — and who should we target with retention spend, given that some customers retain themselves for free?"

`Python` `scikit-learn` `causal ML (X-learner)` `parquet` `config-driven pipelines`

### 🧪 A/B Test Sample Size Calculator
**`experimentation/sample_size_calculator.py`**

Two-sample power analysis for experiment design: minimum detectable effect, baseline rate, power, and significance inputs.

*The decision it supports:* "How long does this test need to run before we can trust it?"

### 💬 LLM Sentiment & Topic Pipeline
**`llm-apps/sentiment_topic_pipeline.py`**

LLM-powered classification of customer reviews into sentiment + extracted topics, with structured JSON output, filterable dashboards, and topic word clouds. Handles CSV/Excel input and exports labeled results.

*The decision it supports:* "What are customers actually complaining about, at scale?"

`OpenAI API` `pandas` `Streamlit` `structured output parsing`

### 📈 Marketing Mix Modeling Assistant
**`marketing-analytics/mmm_assistant.py`**

Upload channel spend + sales data, get an OLS baseline, Lasso-regularized coefficients, and an automated budget reallocation recommendation. Includes plain-English interpretation of every statistical output — built for marketing stakeholders, not statisticians.

*The decision it supports:* "Where should the next marketing dollar go?"

`statsmodels` `scikit-learn` `LassoCV` `Streamlit`

### 💰 Breakeven ROAS Calculator
**`marketing-analytics/breakeven_roas.py`**

Margin-aware ROAS thresholds for paid media. Answers whether a campaign is actually profitable after contribution margin — not just whether the platform dashboard looks good.

*The decision it supports:* "Is this campaign making or losing money?"

### 📊 Ads Measurement & Performance
**`marketing-analytics/ads_measurement.py` · `marketing-analytics/ads_performance.py`**

CLI scripts for channel-level conversion and revenue analysis, and CTR/CPC computation from raw campaign data.

*The decision it supports:* "Which channels convert, and what does a click actually cost us?"

`pandas` `matplotlib` `seaborn`

### 🏘️ Customer Segmentation
**`segmentation/real_estate_segments.py`**

Unsupervised segmentation applied to real estate data — clustering methodology transferable to customer base segmentation.

*The decision it supports:* "Which distinct groups exist in our customer base, and how should we treat them differently?"

---

## Running Any Tool

```bash
pip install -r requirements.txt
```

**Streamlit apps** (open an interactive UI in your browser):

```bash
streamlit run marketing-analytics/mmm_assistant.py
streamlit run marketing-analytics/breakeven_roas.py
streamlit run experimentation/sample_size_calculator.py
streamlit run llm-apps/sentiment_topic_pipeline.py
streamlit run segmentation/real_estate_segments.py
```

LLM-powered tools prompt for an OpenAI API key in the UI — no keys are stored in this repo.

**Plain scripts** (run once, print/plot output — bring your own CSV):

```bash
python marketing-analytics/ads_measurement.py --data path/to/ads_data.csv
python marketing-analytics/ads_performance.py --data path/to/dataset.csv --out processed_dataset.csv
```

`ads_measurement.py` expects columns: `user_id, channel, click_time, conversion_time, revenue`.
`ads_performance.py` expects columns: `clicks, impressions, cost`.
