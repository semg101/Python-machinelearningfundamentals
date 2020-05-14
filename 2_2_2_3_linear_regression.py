import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn import datasets


#Diabetes dataset

#The diabetes dataset consists of 10 physiological variables (age, sex, weight, blood pressure) 
#measure on 442 patients, and an indication of disease progression after one year:

#The task at hand is to predict disease progression from physiological variables

#Load diabetes dataset
diabetes = datasets.load_diabetes()
#Training dataset
diabetes_X_train = diabetes.data[:-20]
diabetes_y_train = diabetes.target[:-20]
#Testing dataset
diabetes_X_test = diabetes.data[-20:]
diabetes_y_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()
print(regr.fit(diabetes_X_train, diabetes_y_train))
print("\n")


print(regr.coef_)
print("\n")

# The mean square error
print(np.mean((regr.predict(diabetes_X_test)-diabetes_y_test) **2))
print("\n")

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
print(regr.score(diabetes_X_test, diabetes_y_test))