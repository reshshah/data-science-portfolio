import streamlit as st

st.set_page_config(page_title="ROAS Calculator", layout="centered")
st.title("📈 ROAS & Media Mix Planner for Small Business")

st.markdown("""
This tool helps you estimate Return on Ad Spend (ROAS) and plan your marketing budget across Upper, Middle & Lower Funnels .
""")

# User inputs
total_budget = st.number_input("Total Monthly Marketing Budget ($)", min_value=100, max_value=100000, value=2000, step=100)

st.subheader("🔧 Budget Allocation & Target ROAS")
col1, col2, col3 = st.columns(3)

with col1:
    sem_pct = st.slider("SEM Budget %", 0, 100, 40)
with col2:
    social_pct = st.slider("Social Media Budget %", 0, 100, 35)
with col3:
    yelp_pct = st.slider("Affiliate Budget %", 0, 100, 15)

display_pct = 100 - (sem_pct + social_pct + aff_pct)
if display_pct < 0:
    st.error("Total budget allocation exceeds 100%. Adjust sliders.")
else:
    st.success(f"Remaining for Awareness/Display: {display_pct}%")

# ROAS targets
sem_roas = st.number_input("Target ROAS for SEM", value=3.5, step=0.1)
social_roas = st.number_input("Target ROAS for Social Media", value=2.0, step=0.1)
aff_roas = st.number_input("Target ROAS for Affiliate", value=1.8, step=0.1)
display_roas = st.number_input("Target ROAS for Display", value=1.5, step=0.1)

# Calculations
channels = ["SEM", "Social Media", "Affiliates", "Display"]
budget_alloc = [sem_pct, social_pct, aff_pct, display_pct]
target_roas = [sem_roas, social_roas, aff_roas, display_roas]
budget_dollars = [round((pct/100)*total_budget, 2) for pct in budget_alloc]
projected_revenue = [round(b * r, 2) for b, r in zip(budget_dollars, target_roas)]

# Output table
import pandas as pd
results_df = pd.DataFrame({
    "Channel": channels,
    "Budget %": budget_alloc,
    "Budget ($)": budget_dollars,
    "Target ROAS": target_roas,
    "Projected Revenue ($)": projected_revenue
})

import ace_tools as tools
tools.display_dataframe_to_user(name="ROAS & Media Mix Plan", dataframe=results_df)

# Total Summary
total_projected = sum(projected_revenue)
avg_roas = round(total_projected / total_budget, 2) if total_budget > 0 else 0

st.subheader("📊 Summary")
st.metric(label="Total Projected Revenue", value=f"${total_projected:,.2f}")
st.metric(label="Average ROAS", value=f"{avg_roas}x")

st.markdown("---")
st.caption("Created for small business owners to plan smarter.")
