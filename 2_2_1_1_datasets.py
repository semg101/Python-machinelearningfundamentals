from sklearn import datasets

import matplotlib.pyplot as plt

import pylab as pl


#Scikit-learn deals with learning information from one or more datasets that are represented as 2D arrays. 
#They can be understood as a list of multi-dimensional observations. 
#We say that the first axis of these arrays is the **samples** axis, while the second is the **features** axis.


#Case 1: iris datasets 

#Load the iris datasets 
iris = datasets.load_iris()
data = iris.data

#The result: (150, 4) It is made of 150 observations of irises, each described by 4 features: 
#their sepal and petal length and width, as detailed in **iris.DESCR**. 
print(data.shape)
print("\n")
print("\n")



#Case 2: digits datasets 

#When the data is not initially in the (n_samples, n_features) shape, 
#it needs to be preprocessed in order to be used by scikit-learn 

#Load the digits datasets 
digits = datasets.load_digits()

#The result: (1797, 8, 8) The digits dataset is made of 1797 8x8 images of hand-written digits 
print(digits.images.shape)
print("\n")


#The result: AxesImages(80,52;496x369.6) <matplotlib.image.AxesImage object at ...> 
print(plt.imshow(digits.images[-1], cmap=plt.cm.gray_r))
print("\n")


#To use this dataset with the scikit, we transform each 8x8 image into a feature vector of length 64 
data = digits.images.reshape((digits.images.shape[0], -1))
print(data)


#To visualize the image 
pl.gray() 
pl.matshow(digits.images[0]) 
pl.show() 


