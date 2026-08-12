"""Marketing Mix Modeling Assistant.

Fits an OLS baseline and a Lasso-regularized model on channel spend vs. sales,
then suggests a budget split based on the Lasso coefficients.

Run with: streamlit run mmm_assistant.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

CHANNELS = ["TV_spend", "Digital_spend", "Social_spend", "Search_spend"]
TARGET = "sales"


def fit_ols(df: pd.DataFrame):
    """OLS baseline with an (unpenalized) intercept."""
    X = sm.add_constant(df[CHANNELS])
    return sm.OLS(df[TARGET], X).fit()


def fit_lasso(df: pd.DataFrame, cv: int = 5, random_state: int = 0):
    """Standardize features, then fit LassoCV.

    Standardizing matters: Lasso penalizes coefficients uniformly, so channels
    on bigger spend scales would otherwise be penalized unfairly. The intercept
    is fit by LassoCV directly and is never penalized (unlike fitting on a
    manually-added constant column).
    """
    X = StandardScaler().fit_transform(df[CHANNELS])
    lasso = LassoCV(cv=cv, random_state=random_state).fit(X, df[TARGET].to_numpy())
    coefs = dict(zip(CHANNELS, lasso.coef_))
    return lasso, coefs


def allocate_budget(coefs: dict, budget: float) -> dict:
    """Split budget proportionally to positive coefficient magnitude.

    A rough heuristic, not an optimization: it ignores saturation and
    diminishing returns. Channels with zero or negative coefficients get
    nothing (spending more on a channel with negative estimated impact
    is never the right call under this model).
    """
    weights = {ch: max(c, 0.0) for ch, c in coefs.items()}
    total = sum(weights.values())
    if total == 0:
        return {ch: 0.0 for ch in coefs}
    return {ch: round(w / total * budget, 2) for ch, w in weights.items()}


def main():
    st.title("Marketing Mix Modeling Assistant")
    st.caption(
        "A deliberately simple MMM: linear, no adstock or saturation curves. "
        "Useful as a first look, not a substitute for a full MMM."
    )

    data_file = st.file_uploader("Upload your marketing data (CSV)", type=["csv"])
    if data_file is None:
        st.info(f"CSV must contain columns: {', '.join(CHANNELS + [TARGET])}")
        return

    df = pd.read_csv(data_file)
    missing = [c for c in CHANNELS + [TARGET] if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.write("### Data Preview")
    st.write(df.head())

    # --- OLS baseline ---
    ols = fit_ols(df)
    st.write("### OLS Baseline")
    st.text(ols.summary())
    st.write(
        "Coefficients estimate each channel's marginal impact on sales, "
        "holding the others constant. Check p-values before trusting any "
        "single coefficient, and remember correlated spend across channels "
        "makes individual estimates unstable."
    )

    # --- Lasso ---
    lasso, coefs = fit_lasso(df)
    st.write("### Lasso Coefficients (standardized features)")
    st.write(coefs)
    st.write(
        "Lasso shrinks weak or redundant channels toward zero, which helps "
        "when channel spends are correlated. A zero here means the channel "
        "added no explanatory power beyond the others — not necessarily that "
        "it drives no sales."
    )

    # --- Budget suggestion ---
    st.write("### Budget Split Suggestion")
    budget = st.number_input("Total budget", min_value=1000, value=5000, step=500)
    allocation = allocate_budget(coefs, budget)
    st.write(allocation)
    st.caption(
        "Proportional to coefficient magnitude — a starting point for "
        "discussion, not an optimized plan. It ignores diminishing returns, "
        "so treat large reallocations with skepticism."
    )


if __name__ == "__main__":
    main()
