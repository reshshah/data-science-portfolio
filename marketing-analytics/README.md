# Marketing Analytics

A small collection of standalone Python scripts exploring common marketing analytics problems: ad performance measurement, budget planning, and marketing mix modeling. These are learning and demonstration projects built on sample or user-supplied data, not production systems.

## Contents

| File | Description |
|---|---|
| `ads_performance.py` | CLI script that computes CTR and CPC from a CSV of clicks, impressions, and cost, plots the CTR distribution, and writes out the processed data. |
| `ads_measurement.py` | CLI script that aggregates conversion rate and revenue by channel from click-level data and visualizes both with bar charts. |
| `breakeven_roas.py` | Streamlit app for planning a media mix across SEM, social, affiliate, and display. Projects revenue from budget allocation and target ROAS inputs. |
| `mmm_assistant.py` | Streamlit app that fits a simple marketing mix model (OLS with Lasso regularization) on channel spend vs. sales, explains the output, and suggests a budget allocation based on coefficient magnitudes. |

## Usage

The two CLI scripts take a `--data` argument pointing to a CSV:

```bash
python ads_performance.py --data your_data.csv        # expects: clicks, impressions, cost
python ads_measurement.py --data your_data.csv        # expects: user_id, channel, click_time, conversion_time, revenue
```

The Streamlit apps run locally:

```bash
streamlit run breakeven_roas.py
streamlit run mmm_assistant.py                         # expects CSV upload with: TV_spend, Digital_spend, Social_spend, Search_spend, sales
```

## Tools

Python (pandas, matplotlib, seaborn, statsmodels, scikit-learn), Streamlit.

## Limitations

- The MMM implementation is intentionally simple: a linear model without adstock, saturation curves, or time effects. Real-world MMM requires those and more.
- Budget allocation in `mmm_assistant.py` is proportional to coefficient magnitude — a rough heuristic, not an optimization.
- No tests or packaging; these are exploratory scripts.
