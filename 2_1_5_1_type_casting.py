import numpy as np
from sklearn import random_projection

#Unless otherwise specified, input will be cast to float64: 
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X = np.array(X, dtype='float32')
X.dtype
print(X.dtype)


#X is float32, which is cast to float64 by fit_transform(X) 
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
X_new.dtype
print(X_new.dtype)
