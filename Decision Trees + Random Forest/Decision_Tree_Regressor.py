import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib as plt

iris = load_iris()
print(iris.data) 