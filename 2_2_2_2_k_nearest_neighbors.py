import numpy as np

from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier


#While experimenting with any learning algorithm, 
#it is important not to test the prediction of an estimator on the data used to fit 
#the estimator as this would not be evaluating the performance of the estimator on new data. 
#This is why datasets are often split into train and test data.

#load the iris datasets 
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target


# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
print(knn.fit(iris_X_train, iris_y_train))
print("\n")

print(knn.predict(iris_X_test))
print("\n")

print(iris_y_test)
