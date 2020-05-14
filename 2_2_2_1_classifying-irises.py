import numpy as np
from sklearn import datasets

#The iris dataset is a classification task consisting in identifying 3 different types of irises #
#(Setosa, Versicolour, and Virginica) from their petal and sepal length and width:#

#load the iris datasets #
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
np.unique(iris_y)
