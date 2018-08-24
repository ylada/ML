# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 08:56:08 2018

Hands on machine Learning with Scikit-Learn and Tensorflow
Chapter 6: Decision Trees
@author: liaoy


------------     Decision Trees & Random Forest     -------------------------
1. Decision Tree: classification, regression, multi-class, multi-output
   can predict class or probability to each class
   sensitive to sample rotation (use PCA to have better orientation)
   sensitive to small variation
   1.1 DecisionTreeClassifier, DecisionTreeREgressor   
       three ways to regulize avoiding overfitting: 
           set max_depth  (default is infinite, 1 is decision stumps) 
           increase: mini_samples_leaf, mini_weight_fraction_leaf
           reduce: max_leaf_nodes, max_features
   1.2 visualize the tree using export_graphviz, export to .dot file
   from sklearn.tree import export_graphviz
2. Random Forest: an essemble of decision trees
   .feature_importance_  gives the importance of all features: root->important
   2.1 Random Forest: tree number as regulation
   rnd_clf = RandomForestClassifier(n_estimators=500,  # # of trees
                    max_leaf_nodes=16,  
                    n_jobs=-1   #multi-gpu
                                 )
3. Boosting: fit the unfitted data
    3.1 AdaBoost
    3.2 Gradient Boost:
    find the right tree number, plot error of staged_predict
    can also do warm_start for early stop checking
    can use subsample=0.25 to train sub-set (25%) of training set
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
    # maximum 120 trees
    gbrt.fit(X_train, y_train)
    errors = [mean_squared_error(y_val, y_pred)
        for y_pred in gbrt.staged_predict(X_val)]
    best_n_estimators = np.argmin(errors)
    gbrt_best = GradientBoostingRegressor(max_depth=2, 
                                          n_estimators=best_n_estimators
                                          )
    print(best_n_estimators)
    plt.plot(errors)
    gbrt_best.fit(X_train, y_train)
    
4. Bagging, Patching, Voting, and Stacking for multiple estimators
   Bagging: training data serve several times for the same estimator
   Patching: training set cannot be re-used for multi-classifiers
   voting: take votes on different classifiers, better than all individuals
   stacking: predict based on the predictions of other classifiers
   
    4.1 Bagging: better than paste
    #some data used multi-times for same estimator, ~37% never used (out of bag)
    # do soft voting if the estimator estimate class probability (predict_proba)
    # use out-of-bag for testing, set oob_score=True
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import accuracy_score
    bag_clf = BaggingClassifier(
                DecisionTreeClassifier(), n_estimators=10,
                max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True
                                )
    bag_clf.fit(X_train, y_train)
    print(bag_clf.oob_score_)    # score of overall out-of-bag data testing
    print(bag_clf.oob_decision_function_)  # score of each out-of-bag instance
    y_pred = bag_clf.predict(X_test)
    accuracy_score(y_test, y_pred)   # accuracy score on test set
    
    4.2  voting: take the estimator with highest probability
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC()
    voting_clf = VotingClassifier(
             estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
             voting='hard'   #'soft' better accuracy but rely on one estimator 
                                  )
    voting_clf.fit(X_train, y_train)
    
    4.3 stacking: add a regression layer ontop of multi-estimators
     sklearn has no stacking, check out 'brew'

"""

from matplotlib import pyplot as plt

# 1.1 DecisionTreeClassifier and visilization using export_graphviz()
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X = iris.data[:, 2:]    #petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

from sklearn.tree import export_graphviz
export_graphviz(
    tree_clf,
    out_file="iris_tree.dot", #image_path("iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True,
                )
# convert .dot to .png:    dot -Tpng tree.dot -o tree.png 
print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))

# 2.1 Random Forests
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500,  ## of trees
                    max_leaf_nodes=16,  
                    n_jobs=-1   #multi-gpu
                                 )
rnd_clf.fit(X, y)
print(rnd_clf.feature_importances_)
#rnd_clf.predict(X_test)

# 3.1 AdaBoost
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5         
                             )
ada_clf.fit(X, y)

# 3.2 Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gbrt = GradientBoostingRegressor(max_depth=2, 
                                 n_estimators=3, 
                                 learning_rate=1.0  #small shrink takes longer
                                 )
gbrt.fit(X, y)
# best tree #? plot error of staged_predict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_val, y_train, y_val = train_test_split(X, y)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120) #max 120 trees
gbrt.fit(X_train, y_train)
errors = [mean_squared_error(y_val, y_pred)
    for y_pred in gbrt.staged_predict(X_val)]
best_n_estimators = np.argmin(errors)
gbrt_best = GradientBoostingRegressor(max_depth=2, 
                                      n_estimators=best_n_estimators
                                      )
print(best_n_estimators)
plt.plot(errors)
gbrt_best.fit(X_train, y_train)

# 4.1 Bagging: better than paste
# some data used multi-times for same estimator, ~37% never used (out of bag)
# specifiy the estimator and number
# do soft voting if the estimator estimate class probability (predict_proba)
# use out-of-bag for testing, set oob_score=True
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
bag_clf = BaggingClassifier(
            DecisionTreeClassifier(), n_estimators=2,
            max_samples=5, bootstrap=True, n_jobs=-1, oob_score=True
                            )
#bag_clf.fit(X_train, y_train)
#print(bag_clf.oob_score_)    # score of overall out-of-bag data testing
#print(bag_clf.oob_decision_function_)  # score of each out-of-bag instance
#y_pred = bag_clf.predict(X_test)
#accuracy_score(y_test, y_pred)   # accuracy score on test set

# 4.2  voting: take the estimator with highest probability
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(
            estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
            voting='hard'   #'soft' better accuracy but rely on one estimator 
                              )
#voting_clf.fit(X_train, y_train)

#4.3 stacking: add a regression layer ontop of multi-estimators
# sklearn has no stacking, check out 'brew'