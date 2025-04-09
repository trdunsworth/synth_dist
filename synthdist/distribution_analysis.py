import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(82169)

# Generate samples from different distributions
n_samples = 10000

# Generate samples
poisson_samples = np.random.poisson(lam=2, size=n_samples)
exponential_samples = np.random.exponential(scale=2, size=n_samples).astype(int)
gamma_samples = np.random.gamma(shape=2, scale=2, size=n_samples).astype(int)
beta_samples = (np.random.beta(a=2, b=5, size=n_samples) * 20).astype(int)
chisquare_samples = (np.random.chisquare(df=2, size=n_samples) * 2).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'Poisson': poisson_samples,
    'Exponential': exponential_samples,
    'Gamma': gamma_samples,
    'Beta': beta_samples,
    'ChiSquare': chisquare_samples,
})

# Calculate summary statistics
def get_distribution_stats(data):
    stats_dict = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Q1': np.percentile(data, 25),
        'Q3': np.percentile(data, 75),
        'Std Dev': np.std(data),
        'Variance': np.var(data),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data)
    }
    return pd.Series(stats_dict)

summary_stats = data.apply(get_distribution_stats)

# Create visualizations
plt.figure(figsize=(15, 15))

# Histograms
for i, col in enumerate(data.columns, 1):
    plt.subplot(3, 2, i)
    sns.histplot(data=data, x=col, kde=True)
    plt.title(f'{col} Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')

plt.tight_layout()
plt.savefig('distribution_histograms.png')
plt.close()

# Box plots
plt.figure(figsize=(12, 6))
sns.boxplot(data=data)
plt.title('Distribution Comparison - Box Plots')
plt.xticks(rotation=45)
plt.savefig('distribution_boxplots.png')
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print(summary_stats.round(4))