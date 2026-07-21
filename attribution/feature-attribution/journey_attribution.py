"""
Bayesian journey-graph attribution on synthetic shopper data.

Pipeline:
1. Simulate customer journeys from a ground-truth first-order Markov process
2. Aggregate to transition counts (the only artifact the model needs - no user IDs)
3. Dirichlet-multinomial posterior over the transition matrix
4. Removal-effect attribution with full posterior uncertainty
5. Compare against last-touch; produce charts

Reproducible: seeded. Requires numpy, pandas, matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- 1. Simulate
STATES = ["Start", "Social", "Display", "Search", "Email", "Direct", "Conv", "Null"]
S = {s: i for i, s in enumerate(STATES)}
CHANNELS = ["Social", "Display", "Search", "Email", "Direct"]

T_TRUE = np.zeros((8, 8))
T_TRUE[S["Start"], [S["Social"], S["Display"], S["Search"], S["Email"], S["Direct"]]] = [
    0.28, 0.16, 0.26, 0.12, 0.18]
T_TRUE[S["Social"]] = [0, 0.06, 0.08, 0.22, 0.16, 0.10, 0.015, 0.365]
T_TRUE[S["Display"]] = [0, 0.07, 0.05, 0.20, 0.10, 0.12, 0.008, 0.452]
T_TRUE[S["Search"]] = [0, 0.04, 0.05, 0.10, 0.12, 0.14, 0.060, 0.490]
T_TRUE[S["Email"]] = [0, 0.05, 0.04, 0.14, 0.08, 0.16, 0.055, 0.475]
T_TRUE[S["Direct"]] = [0, 0.03, 0.03, 0.08, 0.07, 0.06, 0.080, 0.650]
assert np.allclose(T_TRUE[:6].sum(axis=1), 1.0)

N_JOURNEYS = 25_000
MAX_STEPS = 12

journeys = []
for _ in range(N_JOURNEYS):
    path, state = [], S["Start"]
    for _ in range(MAX_STEPS):
        state = rng.choice(8, p=T_TRUE[state])
        if state in (S["Conv"], S["Null"]):
            break
        path.append(STATES[state])
    else:
        state = S["Null"]  # journeys that never resolve are non-converting
    journeys.append((path, STATES[state]))

n_conv = sum(1 for _, o in journeys if o == "Conv")
print(f"journeys={N_JOURNEYS}  conversions={n_conv}  cvr={n_conv/N_JOURNEYS:.4f}")
print(f"avg path length={np.mean([len(p) for p, _ in journeys]):.2f}")

# ------------------------------------------------- 2. Aggregate (privacy layer)
# Everything downstream uses ONLY this count matrix.
C = np.zeros((8, 8), dtype=int)
for path, outcome in journeys:
    seq = ["Start"] + path + [outcome]
    for a, b in zip(seq[:-1], seq[1:]):
        C[S[a], S[b]] += 1
counts = pd.DataFrame(C, index=STATES, columns=STATES)
counts.to_csv("transition_counts.csv")
print("\ntransition counts:\n", counts.iloc[:6])

# Last-touch attribution for comparison
last_touch = {c: 0 for c in CHANNELS}
for path, outcome in journeys:
    if outcome == "Conv" and path:
        last_touch[path[-1]] += 1
lt_share = {c: v / n_conv for c, v in last_touch.items()}
print("\nlast-touch shares:", {k: round(v, 3) for k, v in lt_share.items()})

# --------------------------------------- 3. Dirichlet posterior over the graph
ALPHA = 1.0  # symmetric prior
N_DRAWS = 4000
TRANSIENT = [S["Start"]] + [S[c] for c in CHANNELS]

def p_conversion(T, removed=None):
    """P(absorb in Conv | Start) for transition matrix T, optionally with a
    channel removed (all inflow to it redirected to Null)."""
    T = T.copy()
    if removed is not None:
        r = S[removed]
        T[:, S["Null"]] += T[:, r]
        T[:, r] = 0.0
    Q = T[np.ix_(TRANSIENT, TRANSIENT)]
    R = T[np.ix_(TRANSIENT, [S["Conv"], S["Null"]])]
    absorb = np.linalg.solve(np.eye(len(TRANSIENT)) - Q, R)
    return absorb[0, 0]  # Start row, Conv column

# posterior draws of removal effects and attribution shares
shares = np.zeros((N_DRAWS, len(CHANNELS)))
removals = np.zeros((N_DRAWS, len(CHANNELS)))
base_ps = np.zeros(N_DRAWS)
for d in range(N_DRAWS):
    T = np.zeros((8, 8))
    for i in TRANSIENT:
        T[i] = rng.dirichlet(C[i] + ALPHA)
    p_full = p_conversion(T)
    base_ps[d] = p_full
    re = np.array([max(0.0, 1 - p_conversion(T, c) / p_full) for c in CHANNELS])
    removals[d] = re
    shares[d] = re / re.sum()

# point estimate from raw MLE matrix
T_MLE = np.zeros((8, 8))
for i in TRANSIENT:
    T_MLE[i] = C[i] / C[i].sum()
p_full_mle = p_conversion(T_MLE)
re_mle = np.array([max(0.0, 1 - p_conversion(T_MLE, c) / p_full_mle) for c in CHANNELS])
share_mle = re_mle / re_mle.sum()

print(f"\nbaseline P(conv) MLE={p_full_mle:.4f}  posterior mean={base_ps.mean():.4f}")

rows = []
for j, c in enumerate(CHANNELS):
    lo, med, hi = np.percentile(shares[:, j], [3, 50, 97])
    rows.append({
        "channel": c,
        "last_touch": lt_share[c],
        "markov_mle": share_mle[j],
        "post_median": med, "ci94_lo": lo, "ci94_hi": hi,
        "removal_effect_med": np.median(removals[:, j]),
        "conv_credit_med": med * n_conv,
    })
res = pd.DataFrame(rows).set_index("channel")
pd.set_option("display.width", 160)
print("\n", res.round(4))
res.to_csv("attribution_results.csv")

# P(Social share > Search share) style statements for the article
for a, b in [("Social", "Search"), ("Email", "Display"), ("Social", "Direct")]:
    p = (shares[:, CHANNELS.index(a)] > shares[:, CHANNELS.index(b)]).mean()
    print(f"P(share_{a} > share_{b}) = {p:.3f}")

# deltas vs last touch
for c in CHANNELS:
    d = res.loc[c, "post_median"] - res.loc[c, "last_touch"]
    print(f"{c}: bayes-median minus last-touch = {d:+.3f}")

# ------------------------------------------------------------------ 4. Charts
INK, ACCENT, MUTED, GOLD = "#1a1a2e", "#0f6f6f", "#9aa3ab", "#c9a227"
plt.rcParams.update({
    "font.family": "DejaVu Serif", "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 12})

# fig 2: last-touch vs Bayesian with CI
fig, ax = plt.subplots(figsize=(9.5, 5.4))
x = np.arange(len(CHANNELS))
w = 0.36
ax.bar(x - w / 2, [res.loc[c, "last_touch"] * 100 for c in CHANNELS], w,
       color=MUTED, label="Last-touch", alpha=0.85)
med = [res.loc[c, "post_median"] * 100 for c in CHANNELS]
err_lo = [(res.loc[c, "post_median"] - res.loc[c, "ci94_lo"]) * 100 for c in CHANNELS]
err_hi = [(res.loc[c, "ci94_hi"] - res.loc[c, "post_median"]) * 100 for c in CHANNELS]
ax.bar(x + w / 2, med, w, color=ACCENT, label="Bayesian journey graph (median)")
ax.errorbar(x + w / 2, med, yerr=[err_lo, err_hi], fmt="none",
            ecolor=INK, elinewidth=1.4, capsize=4, label="94% credible interval")
labels = ["Social", "Display", "Paid Search", "Email", "Direct"]
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Share of conversion credit (%)")
ax.set_title("Last-touch vs. Bayesian journey-graph attribution", fontsize=14, pad=12)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("fig2_attribution_comparison.png", dpi=160, bbox_inches="tight")

# fig 3: posterior distributions, contrasting channels
fig, ax = plt.subplots(figsize=(9.5, 5))
for c, col in [("Social", ACCENT), ("Search", GOLD), ("Display", MUTED)]:
    samp = shares[:, CHANNELS.index(c)] * 100
    ax.hist(samp, bins=60, density=True, alpha=0.55, color=col,
            label=f"{'Paid Search' if c == 'Search' else c}")
    ax.axvline(np.median(samp), color=col, lw=1.4, ls="--")
ax.set_xlabel("Share of conversion credit (%)")
ax.set_ylabel("Posterior density")
ax.set_title("What point estimates hide: posterior credit distributions", fontsize=14, pad=12)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("fig3_posteriors.png", dpi=160, bbox_inches="tight")

print("\nsaved: fig2_attribution_comparison.png, fig3_posteriors.png, "
      "transition_counts.csv, attribution_results.csv")
print("(fig1 journey graph + fig4 small-N comparison: run small_sample_uncertainty.py)")
