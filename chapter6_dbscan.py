import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# 1. Generate the dataset
# -----------------------
# make_moons produces two interleaving half-circles, a common toy dataset.
# noise controls how far points deviate from the perfect arcs.
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

# 2. Apply DBSCAN
# ---------------
# DBSCAN groups points that are closely packed and marks outliers as noise.
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# 3. Get clustering results
# -------------------------
clusters = set(labels)
n_clusters = len([c for c in clusters if c != -1])  # '-1' indicates noise
n_noise = list(labels).count(-1)

print("Cluster labels found:", clusters)
print(f"DBSCAN found {n_clusters} clusters and labeled {n_noise} points as noise.\n")

# 4. Plot the original data vs. DBSCAN result
# -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Left subplot: Original Data, colored by true labels (y_true)
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30, edgecolor='k')
axes[0].set_title("Original Data (Colored by True Label)")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

# (b) Right subplot: DBSCAN Clusters
# Create a color map for the different cluster labels
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap('rainbow', len(unique_labels))

for label_val in unique_labels:
    # Select only the points with this label
    mask = (labels == label_val)

    # Noise is labeled as -1 by DBSCAN
    if label_val == -1:
        # Plot noise as black
        axes[1].scatter(X[mask, 0], X[mask, 1], c='black', s=30, edgecolor='k', label='Noise')
    else:
        # Each cluster gets a different color
        axes[1].scatter(X[mask, 0], X[mask, 1], c=[colors(label_val)], s=30, edgecolor='k',
                        label=f'Cluster {label_val}')

axes[1].set_title("DBSCAN Clusters")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")

# (c) Add legend (avoid duplicates by using a set of labels)
handles, labels_legend = axes[1].get_legend_handles_labels()
unique_legend = dict(zip(labels_legend, handles))
axes[1].legend(unique_legend.values(), unique_legend.keys(), loc="upper right")

plt.tight_layout()
plt.show()
