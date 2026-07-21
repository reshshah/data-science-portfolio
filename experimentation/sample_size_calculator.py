# streamlit_sample_size_app.py

import streamlit as st
import math
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

st.set_page_config(page_title="Sample Size Calculator", layout="centered")
st.title("📊 Sample Size Calculator: Two Proportions")

st.markdown("""
This app estimates sample size required per group for comparing two proportions using:
- 🧪 **Chi-squared test** (z-test approximation)
- 🧬 **Fisher’s Exact test** (via simulation)

Useful for any studies where the outcome is binary.
""")

# Inputs
test_type = st.radio("Choose test type", ["Chi-Squared Test", "Fisher's Exact Test (Simulated)"])

p1 = st.number_input("Proportion in Treatment Group ", 0.0, 1.0, 0.6, step=0.01)
p2 = st.number_input("Proportion in Control Group", 0.0, 1.0, 0.3, step=0.01)
alpha = st.slider("Significance Level (α)", 0.001, 0.2, 0.05)
power = st.slider("Power (1 - β)", 0.5, 0.99, 0.8)
dropout = st.slider("Expected Dropout Rate", 0.0, 0.5, 0.15)

st.markdown("---")

if test_type == "Chi-Squared Test":
    st.subheader("📈 Chi-Squared Test Result")

    try:
        effect_size = proportion_effectsize(p1, p2)
        analysis = NormalIndPower()
        n = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1, alternative='two-sided')
        n_adj = math.ceil(n / (1 - dropout))
        total = 2 * n_adj

        st.success(f"Required sample size per group (adjusted for {int(dropout*100)}% dropout): **{n_adj}**")
        st.info(f"Total sample size for the study: **{total}**")

    except Exception as e:
        st.error(f"Error in calculation: {e}")

else:
    st.subheader("🧬 Fisher’s Exact Test Result (Simulated)")

    sims = st.slider("Number of Simulations", 1000, 20000, 5000, step=1000)

    @st.cache_data(show_spinner=True)
    def simulate_fisher(p1, p2, alpha, power_target, sims):
        for n in range(20, 100):
            rejections = 0
            for _ in range(sims):
                group1 = np.random.binomial(1, p1, n)
                group2 = np.random.binomial(1, p2, n)
                table = [[sum(group1), n - sum(group1)],
                         [sum(group2), n - sum(group2)]]
                _, pval = fisher_exact(table)
                if pval < alpha:
                    rejections += 1
            estimated_power = rejections / sims
            if estimated_power >= power_target:
                return n, estimated_power
        return None, None

    with st.spinner("Running simulation..."):
        n_fisher, pwr = simulate_fisher(p1, p2, alpha, power, sims)

    if n_fisher:
        n_adj_fisher = math.ceil(n_fisher / (1 - dropout))
        total_fisher = 2 * n_adj_fisher
        st.success(f"Required sample size per group (adjusted for {int(dropout*100)}% dropout): **{n_adj_fisher}**")
        st.info(f"Total sample size for the study: **{total_fisher}**")
        st.caption(f"(Estimated power from simulation: {pwr:.3f})")
    else:
        st.error("Unable to reach desired power with sample size up to 100 per group. Try lowering the power or alpha.")
