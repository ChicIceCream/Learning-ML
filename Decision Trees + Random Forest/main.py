from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# Load data
data = load_breast_cancer()
X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Decision Tree classifier
clf1 = DecisionTreeClassifier()

def desicionTreeClassifier(X_train, X_test, Y_train, Y_test):
    dt_data = []
    for i in range(20):
        clf1 = DecisionTreeClassifier()
        clf1.fit(X_train, Y_train)
        
        dt_data.append(clf1.score(X_test, Y_test) * 100)
        
        print(f"{i}'th iteration")
        
    return dt_data, clf1

# Random Forest classifier
clf2 = RandomForestClassifier()

def randomForestClassifier(X_train, X_test, Y_train, Y_test):
    rf_data = []
    for i in range(20):
        clf2.fit(X_train, Y_train)
        
        rf_data.append(clf2.score(X_test, Y_test) * 100)  
        
        print(f"{i}'th iteration")
        
    return rf_data

# Get results for Decision Tree
dt_scores, trained_dt_model = desicionTreeClassifier(X_train, X_test, Y_train, Y_test)

# Get results for Random Forest
rf_scores = randomForestClassifier(X_train, X_test, Y_train, Y_test)

# Print the maximum accuracy achieved in both models
print(f'Decision Tree modelling max accuracy: {max(dt_scores)}%')
print(f'Random Forest modelling max accuracy: {max(rf_scores)}%')

# Plot the Decision Tree
plt.figure(figsize=(10, 10))
tree.plot_tree(trained_dt_model, filled=True)
plt.show()

# Note: You cannot directly plot a RandomForestClassifier like a single tree.
# If you want to plot one of the trees in the forest, you can do so as follows:
plt.figure(figsize=(10, 10))
tree.plot_tree(clf2.estimators_[0], filled=True)
plt.show()
