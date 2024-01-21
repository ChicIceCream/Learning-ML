import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

print(data.feature_names)
print(data.target_names)

X_train, X_test, Y_train, Y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, Y_train)

print(clf.score(X_test, Y_test))