import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load the Iris dataset
data = load_iris()
X = data.data
Y = data.target

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and fit the KNeighborsRegressor model
model = KNeighborsRegressor(n_neighbors=4)
model.fit(X_train, Y_train)

# Calculate and print the model score (R^2 score for regression)
score = model.score(X_test, Y_test)
print(f'Accuracy : {score * 100:.2f}%')

# -------------------------
# Visualization 1: PCA Scatter Plot
# -------------------------
# Reduce dimensions to 2 for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Iris Dataset Visualized with PCA')
plt.colorbar(scatter, label='Species')
plt.show()

# -------------------------
# Visualization 2: Predicted vs Actual Values
# -------------------------
# Predict using the test set
Y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, color='blue', edgecolor='k', s=70)
# Diagonal line for reference
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()
