import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset to examine its structure and contents
file_path = 'datasets/11-16-22-mission2.csv'
df = pd.read_csv(file_path)

# Display the first few rows and basic information about the dataset
# df.head(), df.info()

# Step 1: Select numeric columns relevant for PCA
columns_for_pca = [
    'Conductivity (mmhos/cm)', 'Temperature (C)', 'Salinity (ppt)',
    'Sound Speed (m/s)', 'pH', 'Turbid+ NTU', 'Chl ug/L',
    'BGA-PC cells/mL', 'ODOsat %', 'ODO (mg/L)'
]

# Extract these columns
df_pca = df[columns_for_pca]

# Step 2: Verify no missing values in the selected columns
df_pca = df_pca.dropna()

# Step 3: Standardization of the data
scaler = StandardScaler()
df_pca_scaled = scaler.fit_transform(df_pca)

print(df_pca_scaled)

# Step 4: PCA Implementation
pca = PCA(n_components=len(columns_for_pca))
pca_result = pca.fit_transform(df_pca_scaled)

print(pca_result)

# Explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_

print(explained_variance)

# Creating a DataFrame for PCA results for easier interpretation
df_pca_results = pd.DataFrame(data=pca_result,
                              columns=[f'PC{i+1}' for i in range(len(columns_for_pca))])

# Display explained variance and PCA results
print(explained_variance)
print(df_pca_results.head())



# Plotting the explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='blue')
plt.ylabel('Variance Explained')
plt.xlabel('Principal Components')
plt.title('Variance Explained by Each Principal Component')
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(axis='y')
plt.show()

# PCA loading scores (to understand which features contribute to each PC)
loadings = pca.components_

# Plot heatmap of the PCA loadings
plt.figure(figsize=(12, 8))
plt.imshow(loadings, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Loading Scores')
plt.xticks(range(len(columns_for_pca)), columns_for_pca, rotation=45, ha='right')
plt.yticks(range(len(columns_for_pca)), [f'PC{i+1}' for i in range(len(columns_for_pca))])
plt.title('Heatmap of PCA Loadings (Contribution of Each Feature)')
plt.tight_layout()
plt.show()
