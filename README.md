# Applied ML & Analytics Tools

Self-contained tools built from real marketing, growth, and experimentation problems in retail. Each one solves a specific decision a marketing or growth team faces — not a Kaggle exercise.

> Looking for production ML systems work? See my flagship project: [AgentOps — Multi-Agent Churn Prediction](https://github.com/reshshah/agentops-churn-prediction)

---

## Tools

### 📈 Marketing Mix Modeling Assistant
**`marketing-analytics/mmm_assistant.py`**

Upload channel spend + sales data, get an OLS baseline, Lasso-regularized coefficients, and an automated budget reallocation recommendation. Includes plain-English interpretation of every statistical output — built for marketing stakeholders, not statisticians.

*The decision it supports:* "Where should the next marketing dollar go?"

`statsmodels` `scikit-learn` `LassoCV` `Streamlit`

### 💬 LLM Sentiment & Topic Pipeline
**`llm-apps/sentiment_topic_pipeline.py`**

LLM-powered classification of customer reviews into sentiment + extracted topics, with structured JSON output, filterable dashboards, and topic word clouds. Handles CSV/Excel input and exports labeled results.

*The decision it supports:* "What are customers actually complaining about, at scale?"

`OpenAI API` `pandas` `Streamlit` `structured output parsing`

### 💰 Breakeven ROAS Calculator
**`marketing-analytics/breakeven_roas.py`**

Margin-aware ROAS thresholds for paid media. Answers whether a campaign is actually profitable after contribution margin — not just whether the platform dashboard looks good.

*The decision it supports:* "Is this campaign making or losing money?"

### 🧪 A/B Test Sample Size Calculator
**`experimentation/sample_size_calculator.py`**

Two-sample power analysis for experiment design: minimum detectable effect, baseline rate, power, and significance inputs.

*The decision it supports:* "How long does this test need to run before we can trust it?"

### 🏘️ Customer Segmentation
**`segmentation/real_estate_segments.py`**

Unsupervised segmentation applied to real estate data — clustering methodology transferable to customer base segmentation.

---

## Structure

```
├── marketing-analytics/     ← MMM, ROAS, media decisioning
├── llm-apps/                ← LLM-powered analysis tools
├── experimentation/         ← test design & power analysis
├── segmentation/            ← clustering & segmentation
└── requirements.txt
```

## Running Any Tool

```bash
pip install -r requirements.txt
streamlit run <path-to-tool>.py
```

LLM-powered tools prompt for an API key in the UI — no keys are stored in this repo.
