from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

#Multiclass classification makes the assumption that each sample is assigned to one and only one label: 
#a fruit can be either an apple or a pear but not both at the same time. 
#Multilabel classification assigns to each sample a set of target labels. 
#... Multioutput regression assigns each sample a set of target values.

#When using multiclass classifiers, the learning and prediction task that is performed 
#is dependent on the format of the target data fit upon:#
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

#Case 1
#Construction of the estimator
classif = OneVsRestClassifier(estimator=SVC(random_state=0))

#We predict the target data
#the classifier is fit on a 1d array of multiclass labels 
#and the predict() method therefore provides corresponding multiclass predictions.
print(classif.fit(X, y).predict(X))
print("\n")
print("\n")


#Case 2
# the classifier is fit() on a 2d binary label representation of y, using the **LabelBinarizer**
y = LabelBinarizer().fit_transform(y)

#predict() returns a 2d array representing the corresponding multilabel predictions.
#The ouput: the fourth and fifth instances returned all zeroes, 
#indicating that they matched none of the three labels fit upon 
print(classif.fit(X, y).predict(X))
print("\n")
print("\n")



#Case 3
#The classifier is fit upon instances each assigned multiple labels. 
#**The MultiLabelBinarizer** is used to binarize the 2d array of multilabels to fit upon. 
#As a result, predict() returns a 2d array with multiple predicted labels for each instance.
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
print(classif.fit(X, y).predict(X))
