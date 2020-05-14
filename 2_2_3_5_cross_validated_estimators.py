from sklearn import linear_model, datasets



lasso = linear_model.LassoCV()

diabetes = datasets.load_diabetes()

X_diabetes = diabetes.data
y_diabetes = diabetes.target
print(lasso.fit(X_diabetes, y_diabetes))
print("\n")

# The estimator chose automatically its lambda:
print(lasso.alpha_)
