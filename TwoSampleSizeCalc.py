# streamlit_app.py

import streamlit as st
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import math

st.title("📊 Sample Size Calculator for Two Proportions")

st.markdown("""
This calculator estimates the sample size required per group to detect a difference in proportions
using a two-sided test, adjusting for dropout rate.
""")

# User Inputs
p1 = st.number_input("Proportion in Treatment Group (e.g., AIC + SOC)", min_value=0.0, max_value=1.0, value=0.6)
p2 = st.number_input("Proportion in Control Group (e.g., SOC)", min_value=0.0, max_value=1.0, value=0.3)
power = st.slider("Desired Power (1 - Beta)", min_value=0.5, max_value=0.99, value=0.8)
alpha = st.slider("Significance Level (Alpha)", min_value=0.001, max_value=0.2, value=0.05)
dropout_rate = st.slider("Expected Dropout Rate", min_value=0.0, max_value=0.5, value=0.15)

if st.button("Calculate Sample Size"):
    try:
        effect_size = proportion_effectsize(p1, p2)
        analysis = NormalIndPower()
        n_per_group = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1, alternative='two-sided')
        adjusted_n_per_group = math.ceil(n_per_group / (1 - dropout_rate))
        total_n = 2 * adjusted_n_per_group

        st.success(f"✅ Required sample size per group (adjusted): {adjusted_n_per_group}")
        st.info(f"📋 Total sample size for the study: {total_n}")
    except Exception as e:
        st.error(f"⚠️ Error in calculation: {e}")
