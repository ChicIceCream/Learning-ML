from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf1 = DecisionTreeClassifier()
clf1.fit(X_train, Y_train)

clf2 = RandomForestClassifier()
clf2.fit(X_train, Y_train)

print(f'Decision Tree modelling : {clf1.score(X_test, Y_test)*100}')
print(f'Random Forest modelling : {clf2.score(X_test, Y_test)*100}')

