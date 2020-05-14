from sklearn import datasets
from sklearn.svm import SVC

#Regression targets are cast to float64, classification targets are maintained: 

#Load the iris datasets: 
iris = datasets.load_iris()
clf = SVC()


#classification targets are maintained: 
print(clf.fit(iris.data, iris.target))
print("\n")

##The first predict() returns an integer array, since iris.target (an integer array) was used in fit
print(list(clf.predict(iris.data[:3])))
print("\n")
print("\n")





#Regression targets are cast to float64:
print(clf.fit(iris.data, iris.target_names[iris.target]))
print("\n")

##The second predict() returns a string array, since iris.target_names was for fitting: 
print(list(clf.predict(iris.data[:3])))
