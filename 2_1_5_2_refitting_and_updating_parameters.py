import numpy as np
from sklearn.svm import SVC


#Hyper-parameters of an estimator can be updated after it has been constructed via the **sklearn.pipeline.Pipeline.set_params method**. 
#Calling fit() more than once will overwrite what was learned by any previous fit(): 

#**RandomState** exposes a number of methods for generating random numbers drawn from a variety of probability distributions.
rng = np.random.RandomState(0)

#**rand(r, c)** will return an array of r rows and c columns with random values

#an array of 100 rows and 10 columns with random values
X = rng.rand(100, 10)

#binomial(n, p, [size]) 	Draw samples from a binomial distribution. 
#[size] is the number of trials, p is the probability of success and n is the number of successes
y = rng.binomial(1, 0.5, 100)

#an array of 5 rows and 10 clumns with random values
X_test = rng.rand(5, 10)



#Construction of the estimator
clf = SVC()



#The default kernel rbf is first changed to linear after the estimator has been constructed via SVC()

#We call our estimator instance clf, as it is a classifier. 
#It now must be fitted to the model, that is, it must learn from the model. 
#This is done by passing our training set to the fit method. 
print(clf.set_params(kernel='linear').fit(X, y))
print("\n")

#Now you can predict new values, in particular, 
#we can ask to the classifier what is the array which belongs to X_test, 
#which we have not been used to train the classifier: 
print(clf.predict(X_test))
print("\n")
print("\n")



#The default kernel is changed back to rbf to refit the estimator 

#We call our estimator instance clf, as it is a classifier. 
#It now must be fitted to the model, that is, it must learn from the model. 
#This is done by passing our training set to the fit method. 
print(clf.set_params(kernel='rbf').fit(X, y))
print("\n")

#Now you can predict new values, in particular, 
#we can ask to the classifier what is the array which belongs to X_test, 
#which we have not been used to train the classifier: 
print(clf.predict(X_test))
