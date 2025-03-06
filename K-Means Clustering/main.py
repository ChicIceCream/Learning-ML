import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Load the digits dataset
digits = load_digits()

# -------------------------
# Basic Visualization: Sample Digits
# -------------------------
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis('off')
plt.suptitle("Sample Digits from the Dataset")
plt.tight_layout()
plt.show()

# -------------------------
# Clustering: KMeans
# -------------------------
# Scale the data and apply KMeans clustering with 10 clusters
data = scale(digits.data)
model = KMeans(n_clusters=10, init='random', n_init=10, random_state=42)
model.fit(data)

# -------------------------
# Clustering Details: Distribution & Centers
# -------------------------
# Cluster distribution: Count samples in each cluster
labels, counts = np.unique(model.labels_, return_counts=True)
print("Cluster Distribution:")
for cluster, count in zip(labels, counts):
    print(f"Cluster {cluster}: {count} samples")

# Visualize the cluster centers as images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()
for i, ax in enumerate(axes):
    center_img = model.cluster_centers_[i].reshape(8, 8)
    ax.imshow(center_img, cmap='gray')
    ax.set_title(f'Cluster {i}')
    ax.axis('off')
plt.suptitle("Cluster Centers as Images")
plt.tight_layout()
plt.show()

# -------------------------
# Visualization: PCA 2D Scatter Plot with Cluster Labels
# -------------------------
# Reduce dimensions to 2 using PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                      c=model.labels_, cmap='tab10', alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Clustering Results on Digits Dataset")
plt.colorbar(scatter, ticks=range(10))
plt.show()
