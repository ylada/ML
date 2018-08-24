# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 08:21:07 2018
Hands on machine Learning with Scikit-Learn and Tensorflow
Chapter 8: Dimensionality Reduction
@author: liaoy

------------     PCA and LLE     -------------------------
Dimension Reduction while preserving the variance
1. PCA: find hyperplane closest to data, project data to the hyperplane
   principal components are orthogonal, find using singular value decomposition
   PCA assume data are centered, sklearn centers automatically (X = X - X_mean)
   1.1 PCA: set component number or preserved variance, requires whole data
    pca = PCA(n_components = 154)  # or n_compoents = 0.95 for 95% variance
    X_minst_reduced = pca.fit_transform(X_train)
    X_minst_recovered = pca.inverse_transform(X_minst_reduced)

   1.2 Incremetnal PCA: mini-batch PCA
   IncrementalPCA.fit fits all data (on disk),partial_fit fits part in memory
   
"""

# build 3d data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

# 1.1 PCA: dimension reduction, compression
# set component number or set total variance ratio (auto-set component number)
# explained_variance_ratio: variance of each components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)   # or n_components=0.95  min number for 95% variance
#X2D = pca.fit_transform(X)
#print(pca.explained_variance_ratio_)


# image compression, n_component is current dimension if not set
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
mnist = fetch_mldata('MNIST original')
X = mnist["data"]
y = mnist["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)

#pca = PCA()
#pca.fit(X_train)
#cumsum = np.cumsum(pca.explained_variance_ratio_)
#d = np.argmax(cumsum >= 0.95) + 1
#print('d is: ', d)

pca = PCA(n_components = 154)
#X_mnist_reduced = pca.fit_transform(X_train)
#X_mnist_recovered = pca.inverse_transform(X_mnist_reduced)

def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")
    
#plt.figure(figsize=(7, 4))
#plt.subplot(121)
#plot_digits(X_train[::2100])
#plt.title("Original", fontsize=16)
#plt.subplot(122)
#plot_digits(X_mnist_recovered[::2100])
#plt.title("Compressed", fontsize=16)

# 1.2 Incremetnal PCA, mini-batch PCA
# partial_fit() fits each mini batch, split to batches stored in memory
from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):  #split to minibatches
    inc_pca.partial_fit(X_batch)
X_mnist_reduced = inc_pca.transform(X)
# fit() fits whole sample, use np's memmap class for data on disk
#X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m,n))
#batch_size = m // n_batches
#inc_pca.fit(X_mm)

# 1.3 Randomized PCA: faster stochastic algorithm when d is much smaller than n
rnd_pca = PCA(n_components=154, svd_solver="randomized")
#X_reduced = rnd_pca.fit_transform(X)

# 1.4 Kernel PCA: RBF kernel and parameter search
from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
#X_reduced = rbf_pca.fit_transform(X_test)
#grid search for Kernel PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
                ])
param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
               }]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)