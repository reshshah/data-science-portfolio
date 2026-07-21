"""Redraw fig1 with equal aspect + run the 2,000-journey small-data posterior."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

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

N_JOURNEYS, MAX_STEPS = 25_000, 12
journeys = []
for _ in range(N_JOURNEYS):
    path, state = [], S["Start"]
    for _ in range(MAX_STEPS):
        state = rng.choice(8, p=T_TRUE[state])
        if state in (S["Conv"], S["Null"]):
            break
        path.append(STATES[state])
    else:
        state = S["Null"]
    journeys.append((path, STATES[state]))

def count_matrix(js):
    C = np.zeros((8, 8), dtype=int)
    for path, outcome in js:
        seq = ["Start"] + path + [outcome]
        for a, b in zip(seq[:-1], seq[1:]):
            C[S[a], S[b]] += 1
    return C

C = count_matrix(journeys)
TRANSIENT = [S["Start"]] + [S[c] for c in CHANNELS]

def p_conversion(T, removed=None):
    T = T.copy()
    if removed is not None:
        r = S[removed]
        T[:, S["Null"]] += T[:, r]
        T[:, r] = 0.0
    Q = T[np.ix_(TRANSIENT, TRANSIENT)]
    R = T[np.ix_(TRANSIENT, [S["Conv"], S["Null"]])]
    return np.linalg.solve(np.eye(len(TRANSIENT)) - Q, R)[0, 0]

def posterior_shares(C, n_draws=4000, alpha=1.0):
    shares = np.zeros((n_draws, len(CHANNELS)))
    for d in range(n_draws):
        T = np.zeros((8, 8))
        for i in TRANSIENT:
            T[i] = rng.dirichlet(C[i] + alpha)
        p_full = p_conversion(T)
        re = np.array([max(0.0, 1 - p_conversion(T, c) / p_full) for c in CHANNELS])
        shares[d] = re / re.sum()
    return shares

# ---- small-data variant: first 2,000 journeys ("first two weeks")
C_small = count_matrix(journeys[:2000])
n_conv_small = sum(1 for _, o in journeys[:2000] if o == "Conv")
sh_small = posterior_shares(C_small)
sh_full = posterior_shares(C)
print(f"small-N: journeys=2000 conversions={n_conv_small}")
for j, c in enumerate(CHANNELS):
    lo, med, hi = np.percentile(sh_small[:, j] * 100, [3, 50, 97])
    lof, medf, hif = np.percentile(sh_full[:, j] * 100, [3, 50, 97])
    print(f"{c:8s} small: {med:5.1f}% [{lo:5.1f}, {hi:5.1f}]  width={hi-lo:4.1f} | "
          f"full: {medf:5.1f}% [{lof:5.1f}, {hif:5.1f}]  width={hif-lof:4.1f}")
p_disp_beats_social_small = (sh_small[:, 1] > sh_small[:, 0]).mean()
print(f"small-N P(Display > Social) = {p_disp_beats_social_small:.3f}")
print(f"full-N  P(Display > Social) = {(sh_full[:,1] > sh_full[:,0]).mean():.3f}")

# ---- fig 4: CI width comparison small vs full
INK, ACCENT, MUTED, GOLD = "#1a1a2e", "#0f6f6f", "#9aa3ab", "#c9a227"
plt.rcParams.update({
    "font.family": "DejaVu Serif", "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 12})

labels = ["Social", "Display", "Paid Search", "Email", "Direct"]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
y = np.arange(len(CHANNELS))[::-1]
for j in range(len(CHANNELS)):
    lo_s, med_s, hi_s = np.percentile(sh_small[:, j] * 100, [3, 50, 97])
    lo_f, med_f, hi_f = np.percentile(sh_full[:, j] * 100, [3, 50, 97])
    ax.plot([lo_s, hi_s], [y[j] + 0.16] * 2, color=MUTED, lw=5, solid_capstyle="round",
            label="2,000 journeys (wk 1-2)" if j == 0 else None)
    ax.plot(med_s, y[j] + 0.16, "o", color="white", mec=MUTED, mew=1.6, ms=7, zorder=3)
    ax.plot([lo_f, hi_f], [y[j] - 0.16] * 2, color=ACCENT, lw=5, solid_capstyle="round",
            label="25,000 journeys (full quarter)" if j == 0 else None)
    ax.plot(med_f, y[j] - 0.16, "o", color="white", mec=ACCENT, mew=1.6, ms=7, zorder=3)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("Share of conversion credit (%) — 94% credible interval")
ax.set_title("Same model, same channels — only the data volume changed", fontsize=14, pad=12)
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("fig4_uncertainty_shrinks.png", dpi=160, bbox_inches="tight")

# ---- redraw fig1 with equal aspect (true circles, no clipping)
T_MLE = np.zeros((8, 8))
for i in TRANSIENT:
    T_MLE[i] = C[i] / C[i].sum()

fig, ax = plt.subplots(figsize=(11, 6.4))
YS = 2.2  # vertical scale so equal aspect keeps a wide layout
pos = {"Start": (0, 0.5 * YS), "Social": (1.15, 0.88 * YS), "Display": (1.15, 0.12 * YS),
       "Search": (2.3, 0.70 * YS), "Email": (2.3, 0.30 * YS), "Direct": (3.45, 0.5 * YS),
       "Conv": (4.6, 0.74 * YS), "Null": (4.6, 0.26 * YS)}
visits = {s: C[:, S[s]].sum() for s in STATES}
for a in STATES[:6]:
    for b in STATES[1:]:
        pr = T_MLE[S[a], S[b]]
        if pr > 0.055 and a != b:
            x1, y1 = pos[a]; x2, y2 = pos[b]
            col = ACCENT if b == "Conv" else (MUTED if b == "Null" else INK)
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6 + 9 * pr,
                                        color=col, alpha=0.5,
                                        connectionstyle="arc3,rad=0.12"))
for s, (x, y) in pos.items():
    r = 0.30 if s in ("Start", "Conv", "Null") else 0.16 + 0.22 * np.sqrt(
        visits.get(s, N_JOURNEYS) / N_JOURNEYS)
    color = {"Conv": ACCENT, "Null": MUTED, "Start": INK}.get(s, "#ffffff")
    tcolor = "white" if s in ("Conv", "Null", "Start") else INK
    ax.add_patch(plt.Circle((x, y), r, color=color, ec=INK, lw=1.2, zorder=3))
    label = {"Conv": "Purchase", "Null": "No sale", "Search": "Paid\nSearch"}.get(s, s)
    ax.text(x, y, label, ha="center", va="center", fontsize=10, color=tcolor, zorder=4)
ax.set_xlim(-0.5, 5.15); ax.set_ylim(-0.25, YS + 0.25)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("The shopping journey as a graph — fitted transition structure",
             fontsize=14, pad=14)
fig.text(0.5, 0.02, "Edge width = transition probability (edges < 5.5% hidden). "
         "Node size = journey traffic through the channel.",
         ha="center", fontsize=9.5, color=MUTED)
plt.savefig("fig1_journey_graph.png", dpi=160, bbox_inches="tight")
print("saved fig1 (redrawn), fig4_uncertainty_shrinks.png")
