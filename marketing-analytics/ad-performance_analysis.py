
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset.csv")

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
df.to_csv("processed_dataset.csv", index=False)


