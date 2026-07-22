
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="Ad performance: CTR & CPC analysis")
parser.add_argument("--data", default="dataset.csv",
                     help="Path to a CSV with columns: clicks, impressions, cost")
parser.add_argument("--out", default="processed_dataset.csv",
                     help="Path to write the processed CSV with CTR/CPC columns")
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.data)

# Basic Data Overview
print(df.head())
print(df.describe())

# Calculate Click-Through Rate (CTR)
df['CTR'] = df['clicks'] / df['impressions'] * 100

# Calculate Cost Per Click (CPC)
df['CPC'] = df['cost'] / df['clicks']

# Visualize Ad Performance
plt.figure(figsize=(10,5))
sns.histplot(df['CTR'], bins=30, kde=True)
plt.title('Distribution of Click-Through Rate (CTR)')
plt.xlabel('CTR (%)')
plt.ylabel('Frequency')
plt.show()

# Save processed data
df.to_csv(args.out, index=False)


