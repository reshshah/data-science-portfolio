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

## How it works (plain-language)

**The problem:** standard attribution (first-touch, last-touch, linear) just picks a rule for splitting credit across a customer's touchpoints — but never tells you how confident you should be in that split. Teams reallocate ad budgets off single numbers that might just be noise.

**1. Model the journey as a graph (Markov chain)**
Each marketing channel (Social, Display, Paid Search, Email, Direct) is a node. An arrow from Channel A → Channel B represents "customer went from A to B next." There are also two special end states: Purchase and No Sale. This turns a customer's whole journey into a path through a graph, and the whole dataset becomes an 8×8 matrix of "how often does X lead to Y."

**2. Only keep the aggregate counts, not individual paths**
Instead of storing "Customer #4291 went Social→Email→Search→Buy," the model immediately collapses everything into a matrix of counts (how many times did each transition happen, in total). Individual customer paths are thrown away right after counting — that's the privacy angle: nothing downstream ever sees a single person's journey again.

**3. Bayesian uncertainty via a Dirichlet posterior**
Each row of that count matrix (i.e., "from Channel A, where do people go next") is treated as a multinomial distribution. The Dirichlet distribution is the mathematically convenient "conjugate prior" for a multinomial — meaning you can combine a prior belief with observed counts and get an exact answer (Dirichlet(prior + counts)) without needing expensive simulation methods like MCMC. This posterior is then sampled 4,000 times, producing 4,000 slightly different plausible versions of the journey graph rather than one fixed graph.

**4. Removal-effect attribution**
For each of those 4,000 sampled graphs, and for each channel, the method asks: "If I deleted this channel (rerouted its traffic straight to No Sale), how much would the overall conversion rate drop?" That drop is the channel's credit. Doing this across all 4,000 samples gives a distribution of credit values per channel, not just one number — hence "16.8% [16.2%, 17.4%]" instead of just "16.8%."

## Key result

| Channel | Last-touch | Journey graph (median, 94% CI) |
|---|---|---|
| Social | 6.8% | 16.8% [16.2, 17.4] |
| Display | 2.4% | 9.5% [9.2, 9.9] |
| Paid Search | 34.5% | 29.7% [28.7, 30.6] |
| Email | 20.7% | 19.0% [18.1, 19.8] |
| Direct | 35.5% | 25.0% [24.0, 26.1] |

**Takeaway:** last-touch badly shortchanges upper-funnel/awareness channels — Social by 2.5x, Display by 4x — because it only ever credits whatever the customer clicked right before buying, ignoring the earlier touches that put the brand on their radar. Direct gets overcredited, likely because a lot of "direct" traffic only happens because an earlier ad worked.

**On sample size:** with only 2,000 journeys, the credible intervals for Paid Search and Direct overlap — meaning the data can't tell you which one deserves more credit. Only at the full 25,000 journeys do the intervals separate enough to act on. This is the core point: don't reallocate budget until your confidence intervals are actually narrow enough to distinguish the channels.

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

## Code walkthrough

### The "business entities" in the code

The model treats a customer's shopping journey as a walk through a small graph of 8 states:

```python
STATES = ["Start", "Social", "Display", "Search", "Email", "Direct", "Conv", "Null"]
```

- **`Start`** — a pseudo-state everyone begins in
- **`Social`, `Display`, `Search`, `Email`, `Direct`** — the 5 marketing channels (this is the "business entity" that matters — whatever touchpoint categories your company tracks)
- **`Conv`** — the customer purchased (an absorbing/end state)
- **`Null`** — the customer never bought (also an absorbing/end state)

Everything else in the code is math on top of this graph.

### `journey_attribution.py` walkthrough

1. **Fabricate ground truth** (`T_TRUE`, lines 28–36): a hand-picked 8×8 matrix of "if a customer is currently at state A, here's the probability they go to state B next." This is the fake business reality the whole exercise pretends to discover.
2. **Simulate 25,000 customers** (lines 41–51): starting from `Start`, repeatedly roll dice using `T_TRUE` to decide the next touchpoint, stop when they hit `Conv` or `Null` (or after 12 steps, treated as `Null`). Each customer ends up as a list like `["Social", "Email", "Search"] → "Conv"`.
3. **Collapse to a count matrix** (lines 57–66): walk each customer's path as consecutive pairs (`Start→Social`, `Social→Email`, `Email→Search`, `Search→Conv`) and tally how many times each transition happened, into an 8×8 integer matrix `C`. This is the only thing that survives — no individual customer paths are kept past this point, which is the "privacy by design" part.
4. **Also compute last-touch** (lines 68–74), just as the naive baseline to compare against.
5. **Bayesian part** (lines 76–132): for each of the 5 real (non-`Start`) rows of `C`, treat it as a multinomial and draw 4,000 samples from its Dirichlet posterior — i.e., 4,000 slightly different plausible versions of "what are the true transition probabilities out of this channel," given the observed counts. For each sampled matrix, `p_conversion()` solves a linear system (standard absorbing-Markov-chain math) to get the overall probability of ending in `Conv`. Removing a channel means: redirect all its inflow to `Null` and recompute — the resulting drop in conversion probability is that channel's "removal effect." Do this for all 5 channels × 4,000 draws, normalize to shares, and you get a full distribution of credit per channel, not just one number.
6. **Charts** (lines 144–190): bar chart of last-touch vs. Bayesian-with-error-bars, and histograms of the posterior distributions per channel.

### `small_sample_uncertainty.py` walkthrough

Reuses the identical simulation and `p_conversion()` logic, but:
- Runs the same posterior process on **only the first 2,000 journeys** vs. the **full 25,000**, to show the credible intervals shrink as data grows (that's `fig4`).
- Also redraws the journey graph as an actual node-and-arrow diagram (`fig1`) — node size = traffic through that channel, edge width = transition probability.

## Replicating this with real data

The model's only real input is a per-customer, time-ordered sequence of touchpoints ending in a known outcome. So your raw data needs these columns, at minimum:

| Column | Purpose |
|---|---|
| `customer_id` (or `user_id`/anonymous `cookie_id`) | Groups touchpoints into one journey |
| `channel` | Which marketing channel this touchpoint belongs to — you'd map your raw UTM/ad-platform data into a small fixed set of category labels (Social, Display, Search, Email, Direct, etc.) |
| `timestamp` | Orders the touchpoints chronologically within a journey |
| `conversion_flag` or `order_id` / `purchase_timestamp` | Marks whether (and when) that journey ended in a purchase |

From that raw event table, you'd do the same two steps the script does synthetically:

1. **Group by customer**, sort by timestamp, and build each journey as a sequence like `[Search, Email, Direct] → Conv` (or `→ Null` if they never bought within your attribution window — you'd need to pick a lookback window, e.g. 30 or 90 days, same role as `MAX_STEPS=12` here).
2. **Count transitions** between consecutive touchpoints (including `Start→first_channel` and `last_channel→outcome`) into an N×N matrix, where N = number of channels + 3 (Start, Conv, Null).

Everything after that — the Dirichlet posterior, removal effects, credible intervals — is generic and doesn't change; it just operates on whatever count matrix you feed it. Note: this version doesn't use revenue at all, only counts — if you wanted dollar-weighted attribution you'd need a `revenue` column too, but that'd be an extension beyond what's here.

## Limitations

- **First-order Markov assumption**: the model only looks one step back ("what channel did they just come from"), not the full history. A longer memory would be more accurate but risks running out of data to estimate all the extra combinations.
- **Correlational, not causal**: "removal effect" tells you what's associated with conversion, not proven cause-and-effect. Real budget decisions should be validated with actual holdout/incrementality experiments, not just this model.
- **No timing information**: it doesn't account for how long ago a touchpoint happened — a click from 3 months ago is treated the same as one from yesterday.

## Extensions I'd make for production

Higher-order transition memory, recency-decayed edges, differential-privacy noise on the count matrix, and calibration of removal effects against geo-holdout incrementality experiments. See the article's limitations section for why each matters.
