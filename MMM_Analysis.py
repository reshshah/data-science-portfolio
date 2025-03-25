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
        
        st.write("#### Interpretation:")
        st.write("The model summary provides key statistical insights, including R-squared, p-values, and coefficients.")
        st.write("- A higher R-squared value suggests the model explains a large proportion of sales variance.")
        st.write("- Significant p-values (<0.05) indicate strong relationships between variables and sales.")
        st.write("- The coefficients show the impact of each marketing channel on sales.")
        
        # Apply Lasso Regularization
        lasso = LassoCV(cv=5).fit(X, y)
        st.write("### Lasso Regularization Coefficients")
        st.write(dict(zip(X.columns, lasso.coef_)))
        
        st.write("#### Interpretation:")
        st.write("Lasso regularization helps to reduce overfitting and select the most important features.")
        st.write("- Coefficients that are zero indicate that the feature has minimal impact on sales.")
        st.write("- Positive coefficients suggest a positive relationship with sales, while negative ones suggest a diminishing effect.")
        
        # Budget Optimization
        st.write("### Budget Optimization")
        budget = st.number_input("Enter total budget for optimization", min_value=1000, value=5000, step=500)
        
        # Normalize coefficients to allocate budget
        coef_sum = sum(abs(lasso.coef_))
        budget_allocation = {col: abs(coef) / coef_sum * budget for col, coef in zip(X.columns[1:], lasso.coef_[1:])}
        
        st.write("Recommended Budget Allocation:")
        st.write(budget_allocation)
        
        st.write("#### Insights:")
        st.write("The budget allocation suggests where funds should be prioritized based on their effectiveness.")
        st.write("- Higher allocated amounts indicate a strong influence on sales.")
        st.write("- Channels with low allocations may not be as impactful and could be reconsidered.")
        st.write("- Running A/B tests with different budget allocations can further refine effectiveness.")
    else:
        st.write("Error: CSV must contain the required columns: TV_spend, Digital_spend, Social_spend, Search_spend, sales")
