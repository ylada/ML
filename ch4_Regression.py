# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:22:17 2018
Hands on machine Learning with Scikit-Learn and Tensorflow
Chapter 4: Regression (Linear, Polynominal, Logistic, softmax)
chapter 5: SVM classfication and regression (Linear, polynomial, Guassian)
@author: liaoy


------------             Regression and SVM          -------------------------
1. Linear regression computes weighted sum of features, plus a bias (intercept)
   Mean Square Error(MSE) is a convex function no local minina
   learning rate is a factor to multiply the gradient
   the bias and fitted coefficients are: .intercept_ and .coef_
   1.1 Normal Equation
       from sklearn.linear_model import LinearRegression
   1.2 Stochastic Gradient Descent: pick one radom instance to train
       less regular, bounce around minima
       from sklearn.linear_model import SGDRegressor
   1.3 Batch Gradient Descent; mini Batch: No sk-learn solution
   1.4 Polynomial Regression: add powers of each feature, then Linear Regression
   polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("sgd_reg", LinearRegression()),
        ])
   1.5 Regulation: add weighted coeffients to training, but not prediction
       Ridge (l2, 2nd-order):  (l2) is for SGD regression parameter
       Lasso (l1 1st-order): tends to completely remove least important feature
       Elastic Net(l2+l1): better than Lasso
       Ridge is ususally default; Use Elastic when a few features important
   1.6 Early stop, find(clone) right iteration/model for minimum error
       set max_iter to 1 and enable warm start; manually iterate
    from sklearn.base import clone
    sgd_reg = SGDRegressor(max_iter=1, warm_start=True, penalty=None,
                           learning_rate="constant", eta0=0.0005)
    minimum_error = float("inf")
    best_epoch = None
    best_model = None
    all_error = []
    for epoch in range(1000):
        sgd_reg.fit(X, y.ravel())   #continue from it left off
        y_predict = sgd_reg.predict(X)
        val_error = mean_squared_error(y_predict, y)
        all_error.append(val_error)
        if val_error < minimum_error:
            minimum_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg) #clone the best model
       
2. Logistic Regression for classification
    2.1 logistic regression
    turn linear regression to probability w/ sigmoid: p=sig(theta(T) dot x)
    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    2.2 sofmax regression: multiple classes regression, but not multi-output
         simply add multiclass="multinomial" to logistic Regression 
    softmax_reg = LogisticRegression(multi_class="multinomial", 
                                     solver="lbfgs", C=10)
    softmax_reg.fit(X, y)
    
3. SVM: linear or nonlinear classification, regression, or outlier detection
    Feature scale sensitive, so normalization important, 
    SVM does not output probabilities
    three pasckages in sklearn: LinearSVC, SVC, SGDClassifier
    LinearSVC no kernel trick, feature->complexity, handle large training sets
    SVC class do kernel trick: more features w/o complexity, for small training
    3.1 linear SVM classification: sensitive to feature scale
    soft margin C: smaller c => wider street, more margin violations
    svm_clf = Pipeline([
          ("scaler", StandardScaler()),
          ("linear_svc", LinearSVC(C=1, loss="hinge")),
                  ])
    For large datasets, Stochastic Gradient Descent using SVM
    SGDClassifier(loss="hinge", alpha=1/(m*c))  #not so fast as LinearSVC
    3.2 polynomial kernel, not really adding high degree features
        may want to gridsearch best degree and C
    from sklearn.svm import SVC
    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),                            
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))                        
                                ])
    3.3 Gaussian RBF Kernel: good for small size training set
        gamma is regulation, high gamma, irregular boundary, overfitting
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))                               
                                   ])
    3.4 SVM regression: use epcilon as margin to include as much data as it can
    LinearSVR: no kernel trick, feature->complexity, for large training set
    SVM: kernel trick, more feature without complexity, for small training set
    SVM: can do polynimal etc
    from sklearn.svm import LinearSVR
    svm_reg = LinearSVR(epsilon=1.5)
    svm_reg.fit(X, y)
    from sklearn.svm import SVR
    svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg.fit(X, y)


    
"""

import numpy as np
import matplotlib.pyplot as plt


X = 2* np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)
X_new = np.array([[0], [2]])

# 1.1 Linear Regression using Batch Gradient Descent 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()    #X is 2D even data have only 1 feature
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))

# 1.2 Linear REgression using Stochatic Gradient Descent
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)

# 1.4 Polynomial regression
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
# notic polynomial is preprocessing, still use linear regressor
from sklearn.preprocessing import PolynomialFeatures
poly_feature = PolynomialFeatures(degree=2, #power of 2
                                  include_bias=False)
X_poly = poly_feature.fit_transform(X)  #fit_transform()
# add more features, so more columns
print(X[0], X_poly[0])
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

# learning curve: 
# 1) split training set to train and validation sets
# 2) train 1 to m instances
# 3) predict and calculate mean_squared_error of training/validation sets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curve(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    
#lin_reg = LinearRegression()
#plot_learning_curve(lin_reg, X, y)
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("sgd_reg", LinearRegression()),
        ])
#plot_learning_curve(polynomial_regression, X, y)

# 1.5 Regulation: Ridge(l2)  
# Ridge (l2)
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky") # Ridge is square
ridge_reg.fit(X, y)
print(ridge_reg.predict([[1.5]]))
sgd_reg = SGDRegressor(penalty="l2")    #"l2" is Ridge regulation
sgd_reg.fit(X, y.ravel())   #SGDRegressor's y is one dimension: use .ravel()

# Lasso regulation (l1)
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
sgd_reg = SGDRegressor(penalty="l1")    #"l1" is Lasso; first order 
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])

# Elastic Net (l2+l1)
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])

# 1.6 Early stop, find(clone) right iteration/model for minimum error
# set max_iter to 1 and enable warm start; manually iterate
from sklearn.base import clone
sgd_reg = SGDRegressor(max_iter=1, warm_start=True, penalty=None,
                       learning_rate="constant", eta0=0.0005)
minimum_error = float("inf")
best_epoch = None
best_model = None
all_error = []
for epoch in range(1000):
    sgd_reg.fit(X, y.ravel())   #continue from it left off
    y_predict = sgd_reg.predict(X)
    val_error = mean_squared_error(y_predict, y)
    all_error.append(val_error)
    if val_error < minimum_error:
        minimum_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg) #clone the best model
#plt.plot(np.sqrt(all_error), "y-+", linewidth=2, label="epoch")
print("best epoch", best_epoch)

from sklearn import datasets
iris = datasets.load_iris()
#print(list(iris.keys()))
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)
# 2.1 logistic regression: single class model
# .predict([[xxx]]) gives 0 (p<0.5) or 1, .predict_proba() gives probability
# can be regulized by l1 or l2 penalities
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Non Iris-Virginica")

# 2.2 softmax regression: multiple classes regression, but not multi-output
# simply add multiclass="multinomial" to logistic Regression
X = iris["data"][:, (2, 3)]
y = iris["target"]  # target can be 3 classes (1, 2, 3)
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)
print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]))

# 3.1 linear SVM; feature scale sensitive, so normalization important
#   SVM does not output probabilities
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
X = iris["data"][:, (2, 3)]
y = (iris["target"] == 2).astype(np.float64)  # iris-virginica
svm_clf = Pipeline([
          ("scaler", StandardScaler()),
          ("linear_svc", LinearSVC(C=1, loss="hinge")),
                  ])
svm_clf.fit(X, y)
print(svm_clf.predict([[5.5, 1.7]]))

#3.2 polynomial Kernel SVM
# set degree, doesn't actually add any high degree features
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),                            
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))                        
                                ])
poly_kernel_svm_clf.fit(X, y)

# 3.3 Gaussian RBF Kernel: good for small size training set
# gamma is regulation, high gamma, irregular boundary, overfitting
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))                               
                               ])
rbf_kernel_svm_clf.fit(X, y)

# 3.4 SVM regression: use epcilon as margin to include as much data as it can
# LinearSVR or SVM(for polynomial / Guassian )
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)