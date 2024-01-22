# Import libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
Y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf = SVC(kernel='linear', C=3)
clf.fit(X_train, y_train)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Points')

# Import libraries
import matplotlib.pyplot as plt
# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
Y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf = SVC(kernel='linear', C=3)
clf.fit(X_train, y_train)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
# ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap='coolwarm')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Data Points')

plt.show()

print(clf.score(X_test, y_test) * 100)
