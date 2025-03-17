import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../data/ads_data.csv', parse_dates=['click_time', 'conversion_time'])

# Conversion Rate by Channel
conversion_df = df.groupby('channel').agg(
    total_clicks=('user_id', 'count'),
    conversions=('conversion_time', lambda x: x.notnull().sum()),
    total_revenue=('revenue', 'sum')
).reset_index()

conversion_df['conversion_rate'] = (conversion_df['conversions'] / conversion_df['total_clicks']) * 100

# Plot Conversion Rates
plt.figure(figsize=(10,6))
sns.barplot(x='channel', y='conversion_rate', data=conversion_df, palette='Blues_d')
plt.title('Conversion Rate by Channel (%)')
plt.ylabel('Conversion Rate (%)')
plt.xlabel('Channel')
plt.show()

# Plot Total Revenue by Channel
plt.figure(figsize=(10,6))
sns.barplot(x='channel', y='total_revenue', data=conversion_df, palette='Greens_d')
plt.title('Total Revenue by Channel')
plt.ylabel('Revenue')
plt.xlabel('Channel')
plt.show()
