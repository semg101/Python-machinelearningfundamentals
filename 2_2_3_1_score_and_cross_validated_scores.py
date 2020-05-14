from sklearn import datasets, svm
import numpy as np


#The technique used: is called a ***KFold cross-validation***
#First technique

#Load the digits dataset
digits = datasets.load_digits()

X_digits = digits.data
y_digits = digits.target

#Construction of the estimator
svc = svm.SVC(C=1, kernel='linear')

#As we have seen, every estimator exposes a **score** method that can judge the quality of the fit 
#(or the prediction) on new data. **Bigger is better**
print(svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:]))
print("\n")


#To get a better measure of prediction accuracy (which we can use as a proxy for goodness of fit of the model),
#we can successively split the data in **folds** that we use for training and testing:

#Split array X_digits into 3 sub-arrays of equal size.
X_folds = np.array_split(X_digits, 3)
#Split array Y_digits into 3 sub-arrays of equal size.
y_folds = np.array_split(y_digits, 3)

#The method list() takes sequence types and converts them to lists.
#This is used to convert a given tuple into list.
scores = list()

for k in range(3):
    # We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)

    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)

    scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)
