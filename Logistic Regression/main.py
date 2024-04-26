import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris

data = load_iris()

X = data.data
Y = data.target

print(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# model = LogisticRegression()
model = KNeighborsRegressor(n_neighbors=4)

model.fit(X_train, Y_train)

print(f'Accuracy : {(model.score(X_test, Y_test)) * 100}')