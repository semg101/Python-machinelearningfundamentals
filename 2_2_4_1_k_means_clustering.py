from sklearn import cluster, datasets
import numpy as np

#The problem solved in clustering
#Given the iris dataset, if we knew that there were 3 types of iris, 
#but did not have access to a taxonomist to label them: 
#we could try a clustering task: split the observations into well-separated group called clusters.

#Note that there exist a lot of different clustering criteria and associated algorithms. 
#The simplest clustering algorithm is **K-means**.

#Load the iris dataset
iris = datasets.load_iris()

X_iris = iris.data
y_iris = iris.target

k_means = cluster.KMeans(n_clusters=3)

print(k_means.fit(X_iris))
#KMeans(algorithm='auto', copy_x=True, init='k-means++', ...
print("\n")

print(k_means.labels_[::10])
#[1 1 1 1 1 0 0 0 0 0 2 2 2 2 2]
print("\n")

print(y_iris[::10])
#[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2]
print("\n")
print("\n")

#Warning: There is absolutely no guarantee of recovering a ground truth. 
#First, choosing the right number of clusters is hard. 
#Second, the algorithm is sensitive to initialization, and can fall into local minima, 
#although scikitlearn employs several tricks to mitigate this issue.

#Clustering in general and KMeans, in particular, can be seen as 
#a way of choosing a small number of exemplars to compress the information. 
#The problem is sometimes known as vector quantization. 
#For instance, this can be used to posterize an image:
import scipy as sp
try:
    face = sp.face(gray=True)
except AttributeError:
    from scipy import misc
    face = misc.face(gray=True)
X = face.reshape((-1, 1)) # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(n_clusters=5, n_init=1)
print(k_means.fit(X))
print("\n")
#KMeans(algorithm='auto', copy_x=True, init='k-means++', ...
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
face_compressed = np.choose(labels, values)
print(face_compressed)
print("\n")
face_compressed.shape = face.shape
print(face_compressed.shape)