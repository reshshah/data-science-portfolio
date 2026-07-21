# Bayesian Journey-Graph Attribution

Multi-touch attribution that models the customer shopping journey as a Markov graph and quantifies uncertainty with a Dirichlet posterior — so credit estimates come with credible intervals, and the model tells you when the data is sufficient to act.

Companion code for the article: [The shopping journey is a graph](https://metisspace.com/articles/bayesian-journey-graph.html)
Part of the [data-science-portfolio](https://github.com/reshshah/data-science-portfolio) repo — lives in `attribution/`.

## What it does

1. Simulates 25,000 synthetic shopper journeys across five channels from a known ground-truth process (no real customer data anywhere)
2. Aggregates journeys to a transition **count matrix** — the only artifact the model consumes; user-level paths are discarded at this step (privacy by construction)
3. Fits a Dirichlet-multinomial posterior over the journey graph (conjugate — exact, no MCMC)
4. Computes removal-effect attribution across 4,000 posterior draws → credit shares with 94% credible intervals
5. Compares against last-touch, and shows how credible intervals gate budget decisions at small sample sizes

## Key result

| Channel | Last-touch | Journey graph (median, 94% CI) |
|---|---|---|
| Social | 6.8% | 16.8% [16.2, 17.4] |
| Display | 2.4% | 9.5% [9.2, 9.9] |
| Paid Search | 34.5% | 29.7% [28.7, 30.6] |
| Email | 20.7% | 19.0% [18.1, 19.8] |
| Direct | 35.5% | 25.0% [24.0, 26.1] |

Last-touch undervalues upper-funnel channels 2.5–4x and overcredits Direct by 10 points. At 2,000 journeys the Search/Direct ordering is unresolved (overlapping intervals); at 25,000 it separates — the posterior tells you when reallocation is defensible.

## Run it

```bash
pip install numpy pandas matplotlib
python journey_attribution.py        # main analysis: figs 2-3 + CSV outputs
python small_sample_uncertainty.py   # journey graph (fig 1) + small-N comparison (fig 4)
```

Fully seeded and reproducible. Runtime ≈ 30 seconds total.

## Files

```
journey_attribution.py        simulation → counts → posterior → attribution → figs 2-3
small_sample_uncertainty.py   fig 1 journey graph + 2,000 vs 25,000-journey posterior (fig 4)
transition_counts.csv         the aggregate matrix (the model's entire input)
attribution_results.csv       per-channel results with credible intervals
```

## Extensions I'd make for production

Higher-order transition memory, recency-decayed edges, differential-privacy noise on the count matrix, and calibration of removal effects against geo-holdout incrementality experiments. See the article's limitations section for why each matters.
