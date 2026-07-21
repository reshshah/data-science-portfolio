import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV

st.title("Marketing Mix Modeling AI Assistant")

# User input for OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Upload CSV file
data_file = st.file_uploader("Upload your marketing data (CSV)", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)
    st.write("### Data Preview")
    st.write(df.head())
    
    # Check if required columns exist
    required_columns = ["TV_spend", "Digital_spend", "Social_spend", "Search_spend", "sales"]
    if all(col in df.columns for col in required_columns):
        # Define features and target
        X = df[["TV_spend", "Digital_spend", "Social_spend", "Search_spend"]]
        y = df["sales"]
        
        # Add constant for intercept
        X = sm.add_constant(X)
        
        # Fit OLS regression model
        model = sm.OLS(y, X).fit()
        
        st.write("### Model Summary")
        st.text(model.summary())

        # Interpreting the Model Summary
        st.write("""
        **Model Interpretation:**
        - **Coefficients**: The coefficients represent the **estimated impact** of each marketing channel on sales. For example, if the TV_spend coefficient is 0.05, it suggests that for every additional unit spent on TV, sales will increase by 0.05 units, assuming other factors remain constant.
        - **R-squared**: This measures the model's goodness of fit. A higher R-squared (closer to 1) means the model explains a large proportion of the variance in sales.
        - **P-values**: These indicate the statistical significance of each variable. If the p-value is less than 0.05, the variable is statistically significant.
        - **Standard Error**: This measures the variability of the coefficient estimate. A smaller standard error suggests more precise estimates.
        """)

        # Apply Lasso Regularization
        lasso = LassoCV(cv=5).fit(X, y)
        st.write("### Lasso Regularization Coefficients")
        st.write(dict(zip(X.columns, lasso.coef_)))
        
        # Interpreting Lasso Regularization
        st.write("""
        **Lasso Regularization Interpretation:**
        - **Lasso** helps prevent overfitting by shrinking some coefficients to zero. It selects the most important variables and reduces the influence of less relevant ones.
        - The coefficients shown here are the ones **selected by Lasso**, meaning they have the most meaningful impact on sales.
        - A coefficient of zero means that the corresponding marketing channel (e.g., Digital spend) has little to no impact on sales when regularized.
        - Compare the **Lasso coefficients** to the OLS coefficients: if Lasso shrinks a coefficient significantly, it implies the variable was not strongly contributing to the model and might have been overfitting in the original OLS model.
        """)

        # Budget Optimization
        st.write("### Budget Optimization")
        budget = st.number_input("Enter total budget for optimization", min_value=1000, value=5000, step=500)
        
        # Normalize coefficients to allocate budget
        coef_sum = sum(abs(lasso.coef_))
        budget_allocation = {col: abs(coef) / coef_sum * budget for col, coef in zip(X.columns[1:], lasso.coef_[1:])}
        
        st.write("Recommended Budget Allocation:")
        st.write(budget_allocation)
        
        # Interpreting Budget Allocation
        st.write("""
        **Budget Allocation Interpretation:**
        - The model has allocated your marketing budget based on the **relative importance** of each channel's impact on sales.
        - For example, if Lasso assigns a high coefficient to TV_spend, more of the budget should be allocated to TV, as it's the most impactful channel.
        - Use this to **optimize your marketing spend** and focus on the channels that drive the most sales return.
        - Compare these values to historical spending: if you’re already over-allocating to certain channels, consider shifting the budget to others with higher ROI.
        """)

        # Baseline Model for Comparison (no regularization, just OLS)
        baseline_model = sm.OLS(y, X).fit()
        st.write("### Baseline Model Summary (No Regularization)")
        st.text(baseline_model.summary())
        
        st.write("""
        **Baseline Model Interpretation:**
        - This is the original OLS model without any regularization.
        - Compare the baseline coefficients with those from the Lasso model. Significant differences can highlight the channels that Lasso determined as less impactful or prone to overfitting.
        - The **baseline model** provides a starting point, but regularization with Lasso typically leads to better performance and avoids overfitting, especially when dealing with highly correlated features.
        """)
    else:
        st.write("Error: CSV must contain the required columns: TV_spend, Digital_spend, Social_spend, Search_spend, sales")
