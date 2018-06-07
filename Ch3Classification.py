# -*- coding: utf-8 -*-
"""
Created on Fri May 25 08:54:26 2018
Hands on machine Learning with Scikit-Learn and Tensorflow
Chapter 3: Classification using minst
@author: liaoy

------------            Prepare Data          -------------------------
  1. #load data, split to training and testing sets
    1.1  X , y = mnist['data'], mnist['target']
         X_train, X_test = X[:60000], X[60000:]
         y_train, y_test = y[:60000], y[60000:]
    1.2  #shuffle data, avoid sensitive to orders
        shuffle_index = np.random.permutation(60000)
        X_train, y_train = X[shuffle_index], y[shuffle_index]
    1.3 #Optionally: Turn to binary classfications (optional, only if needed)
        y_train_5, y_test_5 = y_train == 5, y_test == 5
    1.4 #optionally: Standarize Data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
-------------             Training            -------------------------
  2. #Optionally split training sets; classify; Evaluate
    2.1  #Pick classifiers: SGDCClassifier, RandomForest
        clf = SGDCClassifier(random_state=42)   
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train_5)      #binary or multiclass
        clf.predict([some_digit_image])  #predict a digit image      
    2.2 #Optional: split training data to n, train n-1, validate the rest 1
        from sklearn.model_selection import StratifiedKFold  #iterate folds
        from sklearn.base import clone    #clone classifier in iteration
        skfolds = StratifiedKFold(n_splits=3, random_state=42)
        for train_index, test_index in skfolds.split(X_train, y_train_5):
            clone_clf = clone(sgd_classifier)
            X_train_folds = X_train[train_index]
            .....
            clone_clf.fit(X_train_folds, y_train_folds)
            y_pred = clone_clf.predict(X_test_fold)
            n_correct = sum(y_pred == y_test_fold)
            print(n_correct / len(y_pred))            
    2.3 #Optional: cross evaluation, check scores
        for scores, find the right threshold for recall, F1, ROC
        from sklearn.model_selection import cross_val_score  #score
        sgd_score = cross_val_score(sgd_classifier, X_train, y_train_5, cv=3, 
                            scoring="accuracy")
        #scoring can be accuracy or other options
    2.4 #Optional: check precision, recall, F1
        #Precision and Recall: Precision=TP/(TP+FP)     recall=TP/(TP+FN)
        #F1 score, harmonic mean: F1=TP/(TP+(FN+FP)/2)
        #detect bad content vidoe: prefer high precision, rejects many good video
        #detct shoplifts: prefer high recall even precision low
        #Some classifiers' decision_function returns decision score on input data
        #can set a threshold for decision scor to filter 
        from sklearn.metrics import precision_score, recall_score, f1_score
        print(precision_score(y_test_fold, y_pred))
        print(recall_score(y_train_5, y_train_pred))
        print(f1_score(y_test_fold, y_pred))
        threshold = 200000
        print(sgd_classifier.decision_function([some_digit])>threshold)
    2.5 #Optional: calulate scores and set threshold to change precision/recall
        #cross_val_predict(method="decision_function") calculate scores, not predict
        #RandomForest has "predict_proba" only for score
        #1st: calculate decision score for all training sets
        #precision can be bumpier with threshold, recall is smooth with threshold
        #2nd: set new threshold (scores>threshold), then get precision/recall
        y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3,
                                     method="decision_function")
        # input 1-d arrays, return precision/recall for different thresholds
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(y_train_5, 
                                                                 y_scores[:,1])
        # plot precision/recalls vs thresholds
        y_train_pred_90 = (y_scores[:,1] > 70000)
        precision_score(y_train_5, y_train_pred_90) 
        #true, binary predict w. threshold
        recall_score(y_train_5, y_train_pred_90) 
        # true, binary predict w. new threshold
    2.6 # Optional:ROC curve, opposite to precicion/recall
        # receiver operating characteristic (ROC) Curve for binary classifiers
        # plot true positive rate (TPR) against false positive rate (FPR)
        # high TRP -> high FPR, choose estimator away from diagonal (random classifier)
        # compare classifier by measure area under the curve (AUC)
        # perfect classifier has AUC = 1, random classifer AUC=0.5
        # precision/recall: when positive is rare, care false postive than false negative
        # ROC: opposite to precision/recall
        # digit 5: few 5s than other digites, so use precision/recall
        from sklearn.metrics import roc_curve, roc_auc_score
        # input binary targets of training set, and scores of training set
        fpr, tpr, thresholds = roc_curve(y_train_5, y_scores[:,1])
        roc_auc_score(y_train_5, y_scores[:,1])
        # plot roc curve (fpr, tpr)
    2.7 # Error analysis
        # Error analysis with training set, confusion matrix between predict & real
        # plot with matshow, then analyze individual 
        y_train_pred = cross_val_predict(sgd_classifier, X_train_scaled, y_train, cv=3)
        # row is actual, column predicted, FN FP TN TP; 2x2
        conf_mx = confusion_matrix(y_train, y_train_pred)
        print(conf_mx)
        row_sums = conf_mx.sum(axis=1, keepdims=True)    #sum of images of each class
        norm_conf_mx = conf_mx / row_sums       #average
        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray) #show where the errors are
        plt.show()
    3. # MultiClass
    3.1. # Multiclass Classification; N classes; Sklearn will auto detect
        # Random Forest, Naive Bayes, can do multiclass
        # SVM, Linear classifers, are binary only
        # one_vs_all is most common (e.g. 5 or not 5), except for SVM, N classify
        # one_vs_one (5 or 1, 5 or 2...) is for SVM, N x(N-1/2)  classify
        # sklearn auto detect and use (oVo or oVo) for binary classifiers
        # oVo: the score of decision_function is n x N array (n training set number)
        sgd_classifier.fit(X_train, y_train)
        print(sgd_classifier.predict([some_digit]))
        some_digit_scores = sgd_classifier.decision_function([some_digit])
        print(np.argmax(some_digit_scores)) #the index of max score
        print(sgd_classifier.classes_)      #the classes to be classified  (0-10)
    3.2 # to force to use one_vs_one for a classifier
        from sklearn.multiclass import OneVsOneClassifier
        ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
        #ovo_clf.fit(X_train, y_train) # ovo_clf.predict([some_digit])
        #len(ovo_clf.estimators_)   #total 45
        cross_val_score(sgd_classifier, X_train, y_train, cv=3, scoring="accuracy")

------------            End of Prepare Data   -------------------------

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

# 1.1 define train and test sets, first 60000 are training sets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 1.2 shuffle the data; avoid sensitivity to the order of training instances
shuffle_index = np.random.permutation(60000)
X_train, y_train = X[shuffle_index], y[shuffle_index]

# 1.3 turn into a binary classifier for digit 5
y_train_5, y_test_5 = (y_train==5), (y_test==5)

# 2.1 train the binary classifier
from sklearn.linear_model import SGDClassifier
sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train_5)
sgd_classifier.predict([some_digit])


# 2.3 Evaluation of training set using StratifiedKFold, split test sets to n
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
    
# 2.4 Evaluate using cross_val_score, not suitable for skewed data
# scoring="accuracy" compares the % of matching classification
from sklearn.model_selection import cross_val_score
sgd_score = cross_val_score(sgd_classifier, X_train, y_train_5, cv=3, 
                            scoring="accuracy")
#print(sgd_score)

# 2.7 evaluate using confusion matrix: find true/false positve/negtive
# first train straftied data and predict the CLEAN fold of training set
# compare with the target of training set
# cros_val_predict(): K-fold cross-valid, returns prediction on each test fold
# cros_val_predict(): K-fold cross-valid, returns evaluation scores
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
#y_train_pred is prediction of each training data, same dimension as y_train
y_train_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3)
#row is actual, column predicted, FN FP TN TP; 2x2
confusionMatrix = confusion_matrix(y_train_5, y_train_pred)

# 2.4 Precision and Recall: Precision=TP/(TP+FP)     recall=TP/(TP+FN)
# F1 score, harmonic mean: F1=TP/(TP+(FN+FP)/2)
# detect bad content vidoe: prefer high precision even rejects many good videos
# detct shoplifts: prefer high recall even precision 
# classifier.decision_function returns decision score on some test data
from sklearn.metrics import precision_score, recall_score, f1_score
print(precision_score(y_test_fold, y_pred))
print(recall_score(y_train_5, y_train_pred))
print(f1_score(y_test_fold, y_pred))
#set a threshold, and compare to decision score, can filter at threshold
threshold = 200000
print(sgd_classifier.decision_function([some_digit])>threshold)

# 2.5 Calulate Scores of test sets, set threshold, and calcuate precision/recall
# First: scores of an estimator, return decision score for all training sets
# cross_al_predict(method="decision_function) to caluate scores, not prediction
# precision can be bumpier with threshold, recall must be smoonth with threshold
# set new threshold by scores > threshold, compare to true to get precision/recall
y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3,
                             method="decision_function")
# inputs are 1-d arrays, return precision/recall for different thresholds
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores[:,1])
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1]) 
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
# set threshold for decision scores, calculate precision/recall of new threshold
y_train_pred_90 = (y_scores[:,1] > 70000)
precision_score(y_train_5, y_train_pred_90) #true, binary predict w. threshold
recall_score(y_train_5, y_train_pred_90) # true, binary predict w. new threshold

# 2.6 receiver operating characteristic (ROC) Curve for binary classifiers
# plot true positive rate (TPR) against false positive rate (FPR)
# high TRP -> high FPR, choose estimator away from diagonal (random classifier)
# compare classifier by measure area under the curve (AUC)
# perfect classifier has AUC = 1, random classifer AUC=0.5
# precision/recall: when positive is rare, care false postive than false negative
# ROC: opposite to precision/recall
# digit 5: few 5s than other digites, so use precision/recall
from sklearn.metrics import roc_curve, roc_auc_score
# input binary targets of training set, and scores of training set
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores[:,1])
roc_auc_score(y_train_5, y_scores[:,1])
def plot_roc_curve(fpr, tpr, label="None"):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
plot_roc_curve(fpr, tpr, "ROC")
plt.show()

# 2.6 Random Forest and others have predict_proba function, not decision_function
# to plot ROC, use proba of positive class as score
from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_classifier, X_train, y_train_5, cv=3,
                                    method="predict_proba")
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, 
                                                      y_probas_forest[:,1])

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()


# 3.1 Multiclass Classification; N classes; Sklearn will auto detect
# Random Forest, Naive Bayes, can do multiclass
# SVM, Linear classifers, are binary only
# one_vs_all is most common (e.g. 5 or not 5), except for SVM, N classify
# one_vs_one (5 or 1, 5 or 2...) is for SVM, N x(N-1/2)  classify
# sklearn auto detect and use (oVo or oVo) for binary classifiers
# oVo: the score of decision_function is n x N array (n training set number)
sgd_classifier.fit(X_train, y_train)
print(sgd_classifier.predict([some_digit]))
some_digit_scores = sgd_classifier.decision_function([some_digit])
print(np.argmax(some_digit_scores)) #the index of max score
print(sgd_classifier.classes_)      #the classes to be classified  (0-10)

# 3.2 to force to use one_vs_one for a classifier
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
#ovo_clf.fit(X_train, y_train) # ovo_clf.predict([some_digit])
#len(ovo_clf.estimators_)   #total 45
cross_val_score(sgd_classifier, X_train, y_train, cv=3, scoring="accuracy")

# 1.4 Standardize the input, even for an image
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_classifier, X_train_scaled, y_train, cv=3,
                scoring="accuracy")

# Error analysis with training set, confusion matrix between predict & real
# plot with matshow, then analyze individual 
y_train_pred = cross_val_predict(sgd_classifier, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)
row_sums = conf_mx.sum(axis=1, keepdims=True)    #sum of images of each class
norm_conf_mx = conf_mx / row_sums       #average
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray) #show where the errors are
plt.show()

# Multilabel: one image has many faces; a number is even/odd and >9
# Multioutput: combination of multiclass and multilabel
# KNN Neighbor can do