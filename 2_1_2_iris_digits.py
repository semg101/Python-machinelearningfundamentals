from sklearn import datasets


#2.1.2 Loading an example dataset


#scikit-learn comes with a few standard datasets
#for instance the iris and digits datasets for classification
#and the boston house prices dataset for regression. 

#load the iris and digits datasets: 
iris = datasets.load_iris()
digits = datasets.load_digits()

#A dataset is a dictionary-like object that holds
#all the data and some metadata about the data

#This data is stored in the .data member, which is a n_samples, n_features array.

#For instance, in the case of the digits dataset, 
#digits.data gives access to the features that can be used to classify the digits samples: 
print(digits.data)
print("\n")

#and digits.target gives the ground truth for the digit dataset, 
#that is the number corresponding to each digit image that we are trying to learn: 
print(digits.target)
print("\n")

#The data is always a 2D array, shape (n_samples, n_features), 
#although the original data may have had a different shape. 
# In the case of the digits, each original sample is 
#an image of shape (8, 8) and can be accessed using: 
print(digits.images[0])