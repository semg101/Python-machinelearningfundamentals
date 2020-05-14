from sklearn.model_selection import KFold, cross_val_score

from sklearn import datasets, svm

# WE get the same result in all of the techniques

#Scikit-learn has a collection of classes which can be used to generate lists of train/test indices for popular crossvalidation strategies.
#Another technique to get the scores like in 2_2_3_1_score_and_validated_scores.py

#Load the digits dataset
digits = datasets.load_digits()

X_digits = digits.data
y_digits = digits.target


#Construction of the estimator
svc = svm.SVC(C=1, kernel='linear')

X = ["a", "a", "b", "c", "c", "c"]
k_fold = KFold(n_splits=3)

#They expose a **split** method which accepts the input dataset to be split 
#and yields the train/test set indices for each iteration of the chosen cross-validation strategy
for train_indices, test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))

print("\n")
print("\n")
#The cross-validation can then be performed easily:
kfold = KFold(n_splits=3)
[print(svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])) for train, test in k_fold.split(X_digits)]

print("\n")
print("\n")
#Another technique
#The cross-validation score can be directly calculated using the **cross_val_score** helper. Given an estimator, 
#the cross-validation object and the input dataset, the **cross_val_score** splits the data repeatedly into a training 
#and a testing set, trains the estimator using the training set and computes the scores based on the testing set
#for each iteration of cross-validation.

#By default the estimator’s score method is used to compute the individual scores.
#Refer the **metrics module** to learn more on the available scoring methods.
#n_jobs=-1 means that the computation will be dispatched on all the CPUs of the computer.
#cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
#array([ 0.93489149, 0.95659432, 0.93989983])

#Alternatively, the scoring argument can be provided to specify an alternative scoring method.
print(cross_val_score(svc, X_digits, y_digits, cv=k_fold, scoring='precision_macro'))
#array([ 0.93969761, 0.95911415, 0.94041254])
