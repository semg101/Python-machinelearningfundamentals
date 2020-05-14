from sklearn import svm
from sklearn import datasets

import pickle

#Construction of the estimator  
clf = svm.SVC()
#Load the iris datasets 
iris = datasets.load_iris()
X, y = iris.data, iris.target


#We call our estimator instance clf, as it is a classifier. 
#It now must be fitted to the model, that is, it must learn from the model. 
#This is done by passing our training set to the fit method. 
print(clf.fit(X, y))
print("\n")

#It is possible to save a model in the scikit 
#by using Python’s built-in persistence model, namely **pickle**: 
s = pickle.dumps(clf)
clf2 = pickle.loads(s)

#Now you can predict new values, in particular, 
#we can ask to the classifier what is the digit of our image in the iris dataset, 
#which we have not been used to train the classifier: 
print(clf2.predict(X[0:1]))
print("\n")

print(y[0])
