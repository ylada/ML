# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:00:34 2018
Hands on machine Learning with Scikit-Learn and Tensorflow
Chapter 9: Tensorflow Basics

@author: liaoy


------------             TensorFlow Basics          -------------------------

"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
w = tf.constant(3)
f = x*x*y + y + 2

init = tf.global_variables_initializer()    #prepare an init node

with tf.Session() as sess:
    init.run()              # run the init to initialize all variables
    # or manually initialize varavaibles by:
    # x.initializer.run()     y.initializer.run()
    result = f.eval()       # functions: evaluate ; variables: initialize


# 1.1 Linear Regression, auto-diff and training operation
# cost function: tf.reduce_mean(tf.square(error), name="mse") #error = y-y_pred
    
# load housing data
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
#housing = fetch_california_housing()
m, n = housing.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
print(scaled_housing_data_plus_bias.mean(axis=0))   #mean of each col
print(scaled_housing_data_plus_bias.mean(axis=1))   #mean of each row
print(scaled_housing_data_plus_bias.mean())
print(scaled_housing_data_plus_bias.shape)

n_epochs = 1000
learning_rate = 0.01
X = tf.constant(scaled_housing_data_plus_bias, tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), tf.float32, name="y")  
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), 
                    name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")  #cost function
gradients = tf.gradients(mse, [theta])[0] #dy/dx_i, x is list, only theta here

training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        #if epoch % 100 == 0:
            #print("epoch", epoch, "MSE=", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()
print("Best Theta:", best_theta)

# 1.2  Optimizer (does auto-diff and updating theta in training)
# 1.2.1 GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        #if epoch % 100 == 0:
            #print("Gradient Descent Optimizer epoch", epoch, "MSE=", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
print("Best Theta using Gradient Descent Optimizer:", best_theta)

# 1.2 placeholder and feed data to algrism
# define placeholders, cost func, learning rate, Optimizer and OP
# initiate variables
# in an iteration: sess.run the OP, feed with training data using placeholders
# evaluate Varaibbles using .eval(), find best parameter

X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), 
                    name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")  #cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices] 
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            save_path = saver.save(sess, "./ch9model.ckpt")   #save training
        for batch_index  in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    print("Best Theta batches with placeholder:", theta.eval())
    save_path = saver.save(sess, "./ch9final_model.ckpt") #save final training
    
with tf.Session() as sess:
    saver.restore(sess, "./ch9final_model.ckpt")    #restore from a model
    best_theta_restored = theta.eval()

# restore graph, and restore tensors in the graph
tf.reset_default_graph()
saver1 = tf.train.import_meta_graph("./ch9final_model.ckpt.meta")
theta1 = tf.get_default_graph().get_tensor_by_name("theta:0")
with tf.Session() as sess:
    saver1.restore(sess, "./ch9final_model.ckpt")
    best_theta_restored = theta1.eval()

# visualize using tensor board
tf.reset_default_graph()
from datetime import datetime

learning_rate = 0.01
X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")  # n is col #
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), 
                    name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y - y_pred
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
init = tf.global_variables_initializer()

# prepare for tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "ch9_log"
logdir = "{}/run-{}".format(root_logdir, now)
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epoches = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)
    
    for batch_index in range(n_batches):
        X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
        if batch_index % 10 == 0:
            summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
            step = epoch * n_batches + batch_index
            file_writer.add_summary(summary_str, step)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    print("best theta in tensorboard: ", theta.eval())

file_writer.close()
# show the tensorboard. go to the command, and run
# tensorboard --logdir C:\Users\liaoy\git\ML\ch9_log\
# following the instructions, open a browser to see graph
