import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

X = np.array([[160,55], [165, 60], [180,75], [185,80], [190,90]])

Z = linkage(X, method='complete', metric='maximum')  # produces the linkage matrix

plt.figure(figsize=(6,4))
dendrogram(Z, labels=np.arange(len(X)))
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Point Index")
plt.ylabel("Distance")
plt.show()