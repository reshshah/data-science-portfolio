"""Breakeven ROAS Calculator & Media Mix Planner.

Margin-aware ROAS thresholds: a campaign is only profitable if its ROAS
exceeds 1 / contribution margin. Plans a budget split across funnels and
flags channels whose target ROAS is below breakeven.

Run with: streamlit run breakeven_roas.py
"""

import pandas as pd
import streamlit as st


def breakeven_roas(contribution_margin: float) -> float:
    """Minimum ROAS at which ad spend stops losing money.

    With contribution margin m, revenue R yields profit m*R - spend.
    Breakeven: ROAS = R / spend = 1 / m.
    """
    if not 0 < contribution_margin <= 1:
        raise ValueError("Contribution margin must be in (0, 1].")
    return round(1.0 / contribution_margin, 2)


def project_revenue(budget_dollars: list, target_roas: list) -> list:
    """Projected revenue per channel = budget x target ROAS."""
    return [round(b * r, 2) for b, r in zip(budget_dollars, target_roas)]


def main():
    st.set_page_config(page_title="ROAS Calculator", layout="centered")
    st.title("📈 Breakeven ROAS & Media Mix Planner")
    st.markdown(
        "Estimate Return on Ad Spend, see the **breakeven ROAS given your "
        "margins**, and plan budget across upper, middle & lower funnel."
    )

    # --- Margin & breakeven ---
    st.subheader("💰 Profitability Threshold")
    margin_pct = st.slider("Contribution margin (%)", 5, 95, 40)
    be = breakeven_roas(margin_pct / 100)
    st.metric("Breakeven ROAS", f"{be}x")
    st.caption(
        f"At a {margin_pct}% contribution margin, any channel returning less "
        f"than ${be:.2f} of revenue per $1 of spend loses money — even if the "
        "platform dashboard shows a 'positive' ROAS."
    )

    # --- Budget allocation ---
    total_budget = st.number_input(
        "Total Monthly Marketing Budget ($)",
        min_value=100, max_value=100000, value=2000, step=100,
    )

    st.subheader("🔧 Budget Allocation & Target ROAS")
    col1, col2, col3 = st.columns(3)
    with col1:
        sem_pct = st.slider("SEM Budget %", 0, 100, 40)
    with col2:
        social_pct = st.slider("Social Media Budget %", 0, 100, 35)
    with col3:
        aff_pct = st.slider("Affiliate Budget %", 0, 100, 15)

    display_pct = 100 - (sem_pct + social_pct + aff_pct)
    if display_pct < 0:
        st.error("Total budget allocation exceeds 100%. Adjust sliders.")
        return
    st.success(f"Remaining for Awareness/Display: {display_pct}%")

    sem_roas = st.number_input("Target ROAS for SEM", value=3.5, step=0.1)
    social_roas = st.number_input("Target ROAS for Social Media", value=2.0, step=0.1)
    aff_roas = st.number_input("Target ROAS for Affiliate", value=1.8, step=0.1)
    display_roas = st.number_input("Target ROAS for Display", value=1.5, step=0.1)

    channels = ["SEM", "Social Media", "Affiliates", "Display"]
    budget_alloc = [sem_pct, social_pct, aff_pct, display_pct]
    target_roas = [sem_roas, social_roas, aff_roas, display_roas]
    budget_dollars = [round((pct / 100) * total_budget, 2) for pct in budget_alloc]
    projected_revenue = project_revenue(budget_dollars, target_roas)

    results_df = pd.DataFrame({
        "Channel": channels,
        "Budget %": budget_alloc,
        "Budget ($)": budget_dollars,
        "Target ROAS": target_roas,
        "Projected Revenue ($)": projected_revenue,
        "Above Breakeven?": ["✅" if r >= be else "❌ losing money" for r in target_roas],
    })

    st.subheader("📊 ROAS & Media Mix Plan")
    st.dataframe(results_df, use_container_width=True)

    below = [ch for ch, r in zip(channels, target_roas) if r < be]
    if below:
        st.warning(
            f"Below breakeven ({be}x) at your margin: {', '.join(below)}. "
            "These channels may still be worth funding for awareness or "
            "incrementality reasons — but not on direct-response economics."
        )

    total_projected = sum(projected_revenue)
    avg_roas = round(total_projected / total_budget, 2) if total_budget > 0 else 0
    st.subheader("📊 Summary")
    st.metric("Total Projected Revenue", f"${total_projected:,.2f}")
    st.metric("Blended ROAS", f"{avg_roas}x", delta=f"{round(avg_roas - be, 2)}x vs breakeven")

    st.markdown("---")
    st.caption("Created for small business owners to plan smarter.")


if __name__ == "__main__":
    main()
