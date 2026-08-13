# Notes: Interpreting Churn & Propensity Model Metrics

Personal reference notes from comparing logistic regression, XGBoost, and
LightGBM on the retail churn and propensity models (`models/train_classifier.py`).

## Test ROC-AUC = 0.797 (logistic regression, churn — the best of the three)

ROC-AUC answers: *if you picked one random customer who actually churned
and one random customer who didn't, how often does the model score the
churner higher?* 0.797 means about 79.7% of the time.

Reference points:
- **0.50** = coin flip, no signal (the dummy baseline)
- **0.70–0.80** = generally the minimum bar for production trust
- **0.797** = right at the top of that range — a genuinely usable churn
  signal, not just "better than random"

## Test PR-AUC = 0.782 (logistic regression, churn)

PR-AUC (precision-recall AUC) is the metric to trust when the positive
class is rare. Unlike ROC-AUC, a random / no-skill model doesn't score 0.5
on PR-AUC — it scores roughly the positive rate itself. Churn is a fairly
common outcome in this dataset (~54% of validation rows, ~51% of test
rows — see the dummy baseline's PR-AUC of 0.542 below), so PR-AUC and
ROC-AUC tell a similar story here; that won't be true for propensity.

## Propensity is a harder, rarer target — read PR-AUC carefully

`propensity_label_30d` is true for only ~18–25% of rows (vs. ~40–54% for
churn), so its dummy-baseline PR-AUC floor is much lower: **0.158** on
validation. The best propensity model (logistic regression) scores
**0.513** validation PR-AUC — about a **3.2x lift** over guessing
randomly, even though its ROC-AUC (0.762) looks superficially similar to
churn's. Comparing propensity's ROC-AUC directly to churn's ROC-AUC
without also checking the PR-AUC floor would be misleading — always check
what the positive rate implies for the "do-nothing" baseline before
judging whether a PR-AUC number is good.

## What actually differs between the three model types

**Logistic Regression** — a linear model. Computes one weighted sum of all
features (`w1*recency_days + w2*tenure_days + ... + b`) and squashes it
into a 0–1 probability via sigmoid. Every feature's effect is independent
and additive — it can't natively express interactions between features
unless you hand-build them. Simple, interpretable (each coefficient has a
direct sign/magnitude meaning), low-capacity.

**XGBoost and LightGBM** — both **gradient-boosted decision trees**,
genuinely different from logistic regression, not just a fancier version
of it. A single decision tree predicts via a sequence of yes/no questions
on features ("is `recency_days` > 30? then is `frequency_90d` < 2?"),
which lets it capture nonlinear relationships and feature interactions
automatically. "Boosting" builds hundreds of small trees *sequentially*,
where each new tree is trained specifically to correct the mistakes of all
trees before it.

**XGBoost vs. LightGBM** differ mainly in *how* they grow each tree:
XGBoost grows level-wise (expands every branch at a given depth before
going deeper — more balanced, more conservative). LightGBM grows leaf-wise
(always splits whichever single leaf reduces error the most next — faster
to fit tightly, but more prone to overfitting on smaller data).

## The throughline for this dataset

Unlike the small-sample synthetic dataset this project started on (~106
unique customers), the retail dataset has **4,996 unique customers and
59,162 training-snapshot rows** — enough that the simpler, lower-capacity
logistic regression model wins outright on both tasks, on every split,
rather than trading off against the tree models. More data narrows the
gap that usually favors gradient boosting; it doesn't always flip the
ranking, but it did here.

## Full metrics comparison

**Churn** (`churn_label_180d`, test set, 9,884 rows, 50.6% positive):

| | Logistic Regression | XGBoost | LightGBM |
|---|---|---|---|
| Test ROC-AUC | **0.797** | 0.782 | 0.776 |
| Test PR-AUC | **0.782** | 0.761 | 0.756 |
| Test Precision | 0.632 | 0.611 | 0.602 |
| Test Recall | 0.906 | 0.935 | 0.945 |
| Test F1 | **0.745** | 0.739 | 0.736 |
| Selected threshold | 0.35 | 0.46 | 0.44 |

**Propensity** (`propensity_label_30d`, test set, 9,884 rows, 18.3% positive):

| | Logistic Regression | XGBoost | LightGBM |
|---|---|---|---|
| Test ROC-AUC | **0.791** | 0.777 | 0.729 |
| Test PR-AUC | **0.513** | 0.493 | 0.466 |
| Test Precision | 0.468 | 0.485 | 0.480 |
| Test Recall | 0.471 | 0.408 | 0.409 |
| Test F1 | **0.469** | 0.443 | 0.441 |
| Selected threshold | 0.63 | 0.50 | 0.50 |

Logistic regression is the best model on both targets, on every metric
that matters, on this dataset. That wasn't true on the earlier synthetic
dataset — see the throughline above for why.

## Cold-start: is there a "new customer" gap here?

The original motivation for the routed churn ensemble (below) was a
cold-start pattern found on the old synthetic dataset: tree models
overfit customers seen during training and did much worse than logistic
regression on genuinely new ones. Re-checked on the retail test set
(9,884 rows: 8,822 for customers also seen in training, 1,062 genuinely
new):

| Model | Overall ROC-AUC | Seen-in-training ROC-AUC (n=8,822) | Genuinely-new ROC-AUC (n=1,062) |
|---|---|---|---|
| Logistic Regression | 0.797 | **0.806** | **0.726** |
| XGBoost | 0.782 | 0.791 | 0.725 |
| LightGBM | 0.776 | 0.785 | 0.726 |

There's still a real gap between seen and new customers (~0.08 ROC-AUC)
for every model, but this time **logistic regression is the best model on
both subgroups**, not just the new-customer one — the pattern that
motivated routing (tree model wins overall but loses badly on new
customers) doesn't reproduce here. All three models land within 0.001 of
each other on genuinely-new customers; there's no segment where a
different model family clearly wins.

## The routed churn ensemble doesn't help on this dataset

`configs/churn_routed_config.yaml` still routes `tenure_days <= 90` to
logistic regression and everyone else to LightGBM (same mechanism as
before — see `src/routing.py`), and `models/predict_churn_routed.py` still
runs correctly against the new data. But since logistic regression is now
the stronger model *everywhere*, not just on the cold-start segment,
routing most of the traffic to LightGBM makes results slightly **worse**
than just using logistic regression for every customer:

| | Logistic Regression | LightGBM | Routed (tenure ≤ 90 → LR) |
|---|---|---|---|
| Test ROC-AUC | **0.797** | 0.776 | 0.773 |
| Test PR-AUC | **0.782** | 0.756 | 0.754 |

**Current recommendation: use plain logistic regression for churn, not the
routed ensemble.** The routing code and config are kept as a working
example of the mechanism — worth revisiting if a future dataset shows the
same cold-start divergence the original synthetic one did, but it isn't
earning its complexity here.

## Is routing the same thing as an ensemble?

No — worth being precise, since the terms mean different things.

**Ensembling** = combining predictions from multiple models *for every
input*, usually by averaging, voting, or a learned combiner (stacking).
Every model contributes to every prediction. Common examples:
- **Bagging** (e.g. Random Forest) — train many models on random subsets,
  average their outputs
- **Boosting** (this is what LightGBM/XGBoost *already are*, internally) —
  train many small trees sequentially, sum their outputs
- **Stacking/blending** — train a meta-model to combine several base
  models' predictions

**Routing** (also called model switching, segmentation, or a
champion/challenger setup) is what `src/routing.py` implements — for any
given customer, exactly **one** model produces the prediction, chosen by a
rule (`tenure_days <= 90`). Logistic regression and LightGBM never both
contribute to the same customer's score. It's a dispatcher, not a blend.

| | Ensembling | Routing |
|---|---|---|
| How many models score each input | All of them | Exactly one |
| How outputs combine | Averaged / voted / stacked | Not combined — a rule picks the winner |
| Goal | Reduce variance, boost accuracy generally | Handle segments best served by different models |

One nuance worth remembering: LightGBM itself **is** already an
ensemble — internally it's hundreds of small decision trees, each
correcting the last, summed together (that's boosting). So routing, when
it *does* help, is routing between one ensemble model and one simple
linear model, not ensembling two things together.

## What happened to the CLV section?

An earlier version of this project had a `future_revenue_180d` regression
target (Ridge/XGBoost/LightGBM regressors, RMSE/MAE/R2/WAPE). It was tied
to the synthetic dataset that's since been replaced with the retail
transaction data, which has no equivalent forward-revenue label defined
yet. `models/train_regressor.py` was removed; the regression machinery in
`src/trainer.py` and `src/evaluator.py` is still there if a revenue target
gets built later. See `MODELING_GUIDE.md`.
