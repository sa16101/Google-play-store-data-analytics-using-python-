"import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
apps_df = pd.read_csv(r"C:\Users\dell\Downloads\googleplaystore.csv")

# Filter for paid apps only
paid_apps = apps_df[apps_df['Type'] == 'Paid']

# Drop missing values in key columns
paid_apps = paid_apps.dropna(subset=['Price', 'Installs', 'Category'])

# Clean and convert data types
paid_apps['Installs'] = paid_apps['Installs'].replace('[+,]', '', regex=True).astype(int)
paid_apps['Price'] = paid_apps['Price'].replace('[$]', '', regex=True).astype(float)

# Calculate Revenue (Installs × Price)
paid_apps['Revenue'] = paid_apps['Installs'] * paid_apps['Price']

# Create scatter plot
plt.figure(figsize=(12, 8))
scatter_plot = sns.scatterplot(
    data=paid_apps,
    x='Installs',
    y='Revenue',
    hue='Category',
    palette='tab10',
    alpha=0.7
)

# Add trendline
sns.regplot(
    data=paid_apps,
    x='Installs',
    y='Revenue',
    scatter=False,
    ax=scatter_plot,
    color='black',
    line_kws={"linewidth": 2}
)

# Final touches
plt.title('Revenue vs. Number of Installs (Paid Apps)', fontsize=16)
plt.xlabel('Number of Installs')
plt.ylabel('Estimated Revenue ($)')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()
