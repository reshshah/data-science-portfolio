# Feature Attribution on Real Clickstream Data

Part 2 of the journey-graph attribution series. Same Bayesian machinery as [Part 1](../README.md) — but on 1.76M real shopping sessions (Retailrocket), attributing purchases to **on-site behaviors** instead of marketing channels, with a formal differential-privacy layer.

Companion code for the article: [What 1.76M real shopping sessions reveal about your site](https://metisspace.com/articles/feature-attribution.html)

## What it does

1. Sessionizes 2.76M real e-commerce events (30-minute inactivity rule) into 1.76M sessions
2. Maps raw events to behavioral states: Browse, Deep Browse, Category Explore, Item Revisit, Add to Cart
3. Aggregates to a transition count matrix — user identifiers are discarded at this step
4. Privacy layer: k-anonymity suppression (k=50) + Laplace noise calibrated for session-level differential privacy (sensitivity 20 via session truncation), with an epsilon sweep
5. Dirichlet-posterior removal effects: the share of purchases that would be lost if each behavior did not exist, with 94% credible intervals

## Key results

| Behavior | Removal effect (94% CI) |
|---|---|
| Browse | 82.8% [82.4, 83.2] |
| Add to Cart | 66.1% [65.4, 66.8] |
| Item Revisit | 33.2% [32.7, 33.7] |
| Deep Browse | 16.3% [16.0, 16.6] |
| Category Explore | 16.2% [15.9, 16.5] |

- **Item Revisit is the hidden workhorse**: a third of purchases depend on shoppers returning to an item they already viewed — twice the load of deep browsing or category exploration.
- **One ordering stays unresolved even at 1.76M sessions**: P(Deep Browse > Category Explore) = 0.73 — not decision-grade.
- **The privacy tax is ~zero**: removal effects are unchanged down to ε = 0.5 and shift only 2–3 points at an aggressive ε = 0.05.

## Run it

```bash
pip install numpy pandas matplotlib pyarrow

# Download the Retailrocket dataset from Kaggle (free account required):
# https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
# Unzip into ./data/  (events.csv, item_properties_part1.csv, item_properties_part2.csv)

python 01_category_map.py     # item -> category lookup from item properties
python 02_sessionize.py       # sessionize + behavioral states -> count matrix
python 03_attribution_dp.py   # k-anonymity + DP sweep + posterior removal effects
python 04_figures.py          # all three figures
```

Seeded and reproducible. Total runtime ≈ 2 minutes. The raw data is **not** committed here — download it from Kaggle and see the dataset page for its license terms; results derived here are aggregate statistics only.

## Data caveat

~35% of recorded purchases have no preceding add-to-cart event in this dataset (5,039 of 14,297 purchases enter the graph from non-cart states) — a known tracking gap in the source data, not a real shopper behavior. Interpret the Add-to-Cart removal effect as a lower bound.

## Files

```
01_category_map.py       item properties -> item_category.csv
02_sessionize.py         events -> behavioral states -> rr_transition_counts.csv
03_attribution_dp.py     privacy layer + posterior -> rr_attribution_results.csv
04_figures.py            behavior graph, removal-effect forest, privacy-tax sweep
```
