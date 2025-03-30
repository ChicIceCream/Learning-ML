import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = list(iris.target_names)  # Convert to list

# -------------------------
# Visualization 1: Basic Scatter Plot (Sepal Length vs Sepal Width)
# -------------------------
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=80)
plt.xlabel(feature_names[0].capitalize())
plt.ylabel(feature_names[1].capitalize())
plt.title("Iris Dataset: Sepal Length vs Sepal Width")
# Use list(target_names) to ensure it's a list of strings
handles, _ = scatter.legend_elements()
plt.legend(handles=handles, labels=target_names, title="Species")
plt.show()

# -------------------------
# Clustering: KMeans on Iris Data
# -------------------------
# Scale the data for clustering
X_scaled = scale(X)
kmeans = KMeans(n_clusters=3, init='random', n_init=10, random_state=42)
kmeans.fit(X_scaled)

# -------------------------
# Visualization 2: Cluster Distribution Histogram
# -------------------------
labels, counts = np.unique(kmeans.labels_, return_counts=True)
print("Cluster Distribution:")
for cluster, count in zip(labels, counts):
    print(f"Cluster {cluster}: {count} samples")
# -------------------------
# Visualization 3: PCA 2D Scatter Plot with Cluster Labels and Centers
# -------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Project cluster centers into the PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                        c=kmeans.labels_, cmap='viridis', edgecolor='k', alpha=0.7)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X', label='Centers')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA 2D Scatter Plot of Iris Data with KMeans Clusters")
plt.legend()
plt.colorbar(scatter, ticks=range(3), label='Cluster Label')
plt.show()
