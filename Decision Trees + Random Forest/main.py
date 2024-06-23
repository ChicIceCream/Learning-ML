from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

X = data.data
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

def desicionTreeClassifier(X_train, X_test, Y_train, Y_test):
    dt_data = []
    for i in range(20):
        clf1 = DecisionTreeClassifier()
        clf1.fit(X_train, Y_train)
        
        dt_data.append(clf1.score(X_test, Y_test)*100)
        
        print(f"{i}'th iteration")
        
    return dt_data

def randomForestClassifier(X_train, X_test, Y_train, Y_test):
    rf_data = []
    for i in range(20):
        clf2 = RandomForestClassifier()
        clf2.fit(X_train, Y_train)
        
        rf_data.append(clf2.score(X_test, Y_test)*100)  
        
        print(f"{i}'th iteration")
        
    return rf_data

print(f'Decision Tree modelling : {max(desicionTreeClassifier(X_train, X_test, Y_train, Y_test))}')
print(f'Random Forest modelling : {max(randomForestClassifier(X_train, X_test, Y_train, Y_test))}')
