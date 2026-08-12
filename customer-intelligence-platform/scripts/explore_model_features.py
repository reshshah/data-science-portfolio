import pandas as pd

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)

df = pd.read_csv("data/processed/customer_model_features.csv")

print("Shape:", df.shape)
print()
print("Dtypes:")
print(df.dtypes)
print()
print("Missing values (columns with any):")
print(df.isna().sum()[df.isna().sum() > 0])
print()
print("Label distribution - churn_label_180d:")
print(df["churn_label_180d"].value_counts())
print()
print("Label distribution - purchase_propensity_label_30d:")
print(df["purchase_propensity_label_30d"].value_counts())
print()
print("Head:")
print(df.head())
