import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5]).reshape(-1,1)
scores = np.array([56, 83, 47, 93, 47, 82, 45, 78, 55, 67, 57, 4]).reshape(-1,1)

model = LinearRegression()
model.fit(time_studied, scores)

print(model.predict(np.array([56]).reshape(-1,1)))

plt.scatter(time_studied, scores)
plt.plot(np.linspace(0,70,100).reshape(-1,1), model.predict(np.linspace(0,70,100).reshape(-1,1)), 'r')
plt.ylim(0,100)
plt.show()

# data = pd.read_csv('student-mat.csv', sep=';')

# data = data[['age', 'sex', 'studytime', 'absences',
#             'G1', 'G2', 'G3']]
# data['sex'] = data['sex'].map({'F' : 0, 'M' : 1})

# prediction = 'G3'

# X = np.array(data.drop([prediction], 1))
# Y = np.array(data[prediction])

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)

# model1 = LinearRegression()

# model1.fit(X_train, Y_train)

# accuracy = model1.score(X_test, Y_test)
# print(accuracy * 100)