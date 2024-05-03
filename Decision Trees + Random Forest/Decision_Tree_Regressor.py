import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

iris = load_iris()
# print(iris) 
# print(iris.data) 

classifier = DecisionTreeClassifier()
classifier.fit(iris.data, iris.target)

plt.figure(figsize=(10,10))
tree.plot_tree(classifier, filled=True)
plt.show()