import plotly.express as px
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


df = pd.read_csv('datasets/biscayne_bay_water_quality2.csv')
print("Features: \n",df.columns)
print("Top 5 rows: \n",df.head())
print("Sum of Null Values: \n",df.isnull().sum())
print("Sum of duplicate Values: \n",df.duplicated().sum())
print("Descriptive Stats: \n",df.describe())

print("Mean Temperature: \n", df["Temperature (c)"].mean())
print("Median Temperature: \n", df["Temperature (c)"].median())
print("Mode Temperature: \n", df["Temperature (c)"].mode())

print("Variance Temperature: \n", df["Temperature (c)"].var())
print("Standard Deviation Temperature: \n", df["Temperature (c)"].std())

# Selecting numerical columns for analysis
numerical_cols = ["Total Water Column (m)",
                  "Salinity (ppt)",
                  "Temperature (c)",
                  "pH",
                  "ODO mg/L"]


# Plot box plots
for col in numerical_cols:
    fig = px.box(df, y=col, title=f"Box Plot of {col}", labels={col: col})
    fig.show()


# Function to detect outliers using the 1.5*IQR rule
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check for outliers in Total Water Column, Temperature, Salinity, pH, and ODO
outlier_dict = {}
for col in numerical_cols:
    outliers, lower, upper = detect_outliers(df, col)
    outlier_dict[col] = {
        "Lower Bound": lower,
        "Upper Bound": upper,
        "Number of Outliers": len(outliers),
        "Outlier Percentage": len(outliers) / len(df) * 100,
    }

# Convert outlier data to a DataFrame and display
outlier_df = pd.DataFrame.from_dict(outlier_dict, orient="index")
print(outlier_df)

# Remove Outliers
df_no_outliers = df.copy()
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]

print("Data After Removing Outliers: \n", df_no_outliers)
#df_no_outliers.to_csv("datasets/bb2_no_outliers.csv")


for column in numerical_cols:
    fig = px.box(df_no_outliers, y=column, title=f"Box Plot of {column}", labels={column: column})
    fig.show()


# Plot histograms
for col in numerical_cols:
    fig = px.histogram(df_no_outliers, x=col, nbins=30, title=f"Histogram of {col}", labels={col: col})
    fig.show()


# Skewness
skewness_values = {}
for col in numerical_cols:
    gp = 3 * (df_no_outliers[col].mean() - df_no_outliers[col].median()) / df_no_outliers[col].std()
    skewness_values[col]=gp
    print(f"Pearson median skew ({col}):", gp)

"""
Using seaborn to visualize histograms for the key variables, 
with skewness values annotated:

- The red dashed line represents the median.
- The blue dashed line represents the mean.
- The skewness value is displayed in the title of each histogram.

Observations:

- Salinity is close to symmetric, with mean and median nearly aligned.
- Temperature and ODO show noticeable right skewness, with the mean pulled to the right of the median.
- pH has mild right skewness, but its mean and median are relatively close

import seaborn as sns
import matplotlib.pyplot as plt

# Set up the layout for histograms with skewness annotations
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

for ax, col in zip(axes.flatten(), numerical_cols):
    sns.histplot(df_no_outliers[col].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title(f"Histogram of {col}\nSkewness: {skewness_values[col]:.2f}")
    ax.axvline(df_no_outliers[col].median(), color='red', linestyle='dashed', label='Median')
    ax.axvline(df_no_outliers[col].mean(), color='blue', linestyle='dashed', label='Mean')
    ax.legend()

plt.tight_layout()
plt.show()
"""

### 3.4 - Estimation
# Calculate sample mean and variance for Salinity, Temperature, and ODO
mean_sal = df_no_outliers['Salinity (ppt)'].mean()
var_sal  = df_no_outliers['Salinity (ppt)'].var()      # by default, sample variance (ddof=1)
std_sal  = df_no_outliers['Salinity (ppt)'].std()      # sample standard deviation

mean_temp = df_no_outliers['Temperature (c)'].mean()
var_temp  = df_no_outliers['Temperature (c)'].var()
std_temp  = df_no_outliers['Temperature (c)'].std()

mean_odo = df_no_outliers['ODO mg/L'].mean()
var_odo  = df_no_outliers['ODO mg/L'].var()
std_odo  = df_no_outliers['ODO mg/L'].std()

print(f"Salinity: mean={mean_sal:.2f}, variance={var_sal:.2f}, std={std_sal:.2f}")
print(f"Temperature: mean={mean_temp:.2f}, variance={var_temp:.2f}, std={std_temp:.2f}")
print(f"Dissolved O2: mean={mean_odo:.2f}, variance={var_odo:.2f}, std={std_odo:.2f}")

# Compute z-scores for each observation in Salinity, Temperature, and ODO
df_no_outliers['Salinity_z']   = (df_no_outliers['Salinity (ppt)']   - mean_sal)  / std_sal
df_no_outliers['Temperature_z'] = (df_no_outliers['Temperature (c)'] - mean_temp) / std_temp
df_no_outliers['ODO_z']        = (df_no_outliers['ODO mg/L']        - mean_odo)  / std_odo

# Look at the first few values to see the z-scores alongside original data
print(df_no_outliers[['Salinity (ppt)', 'Salinity_z', 'Temperature (c)', 'Temperature_z', 'ODO mg/L', 'ODO_z']].head())

### 3.4.2 0 Covariance

# Covariance between Temperature and Salinity
cov_matrix = df_no_outliers[['Temperature (c)', 'Salinity (ppt)']].cov()
cov_temp_sal = cov_matrix.loc['Temperature (c)', 'Salinity (ppt)']
print(cov_matrix)

# Pearson and Spearman correlations between Temperature and Salinity
pearson_temp_sal  = df_no_outliers['Temperature (c)'].corr(df['Salinity (ppt)'], method='pearson')
spearman_temp_sal = df_no_outliers['Temperature (c)'].corr(df['Salinity (ppt)'], method='spearman')

print(f"Covariance(Temperature, Salinity) = {cov_temp_sal:.2f}")
print(f"Pearson r(Temperature, Salinity) = {pearson_temp_sal:.2f}")
print(f"Spearman ρ(Temperature, Salinity) = {spearman_temp_sal:.2f}")


fig = px.imshow(df_no_outliers[numerical_cols].corr(), text_auto=True, aspect="auto")
fig.show()

# Pearson correlation (default)
pearson_corr = df_no_outliers[numerical_cols].corr()
print("Pearson Correlation:\n", pearson_corr)

# Spearman correlation
spearman_corr = df_no_outliers[numerical_cols].corr(method='spearman')
print("\nSpearman Correlation:\n", spearman_corr)

# Kendall correlation
kendall_corr = df_no_outliers[numerical_cols].corr(method='kendall')
print("\nKendall Correlation:\n", kendall_corr)
