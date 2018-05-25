# -*- coding: utf-8 -*-
"""
Created on Fri May 25 08:54:26 2018
Hands on machine Learning with Scikit-Learn and Tensorflow
Chapter 3: Classification using minst
@author: liaoy
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# load mnist data, save to local cache
from sklearn.datasets import fetch_mldata
custom_data_home = 'Ch3Mnist' #save to local cache
#mnist is dict, has 'data', and 'target'
#mnist = fetch_mldata('MNIST original', data_home=custom_data_home)

# inspect data shape, plot a digit
X , y = mnist['data'], mnist['target']
#print(X.shape, y.shape)
some_digit = X[36000]
some_digit_image = X[36000].reshape(28, 28)
#plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
#plt.axis("off")
#plt.show()
#print(y[36000])

# define train and test sets, first 60000 are training sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#shuffle the data; avoid sensitivity to the order of training instances
shuffle_index = np.random.permutation(60000)
X_train, y_train = X[shuffle_index], y[shuffle_index]

#turn into a binary classifier for digit 5
y_train_5, y_test_5 = (y_train==5), (y_test==5)

#train the binary classifier
from sklearn.linear_model import SGDClassifier
sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train_5)
sgd_classifier.predict([some_digit])


# Evaluation of training set using StratifiedKFold, split test sets to n
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)
# split to n folds, interate, train n-1 fold, test on the rest fold
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_classifier)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    print("index: \t", test_index)
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    # all above are using training set (even for X_test)
    # after training, using test set for prediction and evaluation
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
    
# Evaluate using cross_val_score, not suitable for skewed data
# scoring="accuracy" compares the % of matching classification
from sklearn.model_selection import cross_val_score
sgd_score = cross_val_score(sgd_classifier, X_train, y_train_5, cv=3, 
                            scoring="accuracy")
#print(sgd_score)

# evaluate using confusion matrix: find true/false positve/negtive
# first train straftied data and predict the CLEAN fold of training set
# compare with the target of training set
from 
