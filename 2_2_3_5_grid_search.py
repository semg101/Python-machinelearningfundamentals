from sklearn import linear_model, datasets
lasso = linear_model.LassoCV()
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target
print(lasso.fit(X_diabetes, y_diabetes))
#LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
#precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
#verbose=False)
# The estimator chose automatically its lambda:
print(lasso.alpha_)
#0.01229...