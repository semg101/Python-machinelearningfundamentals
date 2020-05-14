from sklearn import linear_model

#LinearRegression will take in its fit method arrays X, y
#and will store the coefficients 𝑥 of the linear model in its coef_ member

reg = linear_model.LinearRegression()
print(reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2]))
print("\n")

#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print(reg.coef_)
#array([ 0.5, 0.5])