# -*- coding: UTF-8 -*-
"""
Created on 2017-06-16

@author: zzx
"""

import numpy as np


def sigmoid(z):
    return 1./(1 + np.exp(-z))


def cost_function(X, y, theta):
    [m, n] = X.shape
    h = sigmoid(np.sum(theta*X, 1))
    cost = -np.mean(y*np.log(h) + (1-y)*np.log(1-h))
    grad = np.ndarray(shape=theta.shape)
    for j in range(n):
        grad[j] = -np.sum((y-h)*X[:, j])/m
    return [cost, grad]


def grad_descent(X, y, init_theta=None, alpha=0.01, inter=100):
    [m, n] = X.shape
    if init_theta is None:
        init_theta = np.zeros(shape=[n])
    theta = init_theta
    for i in range(inter):
        [cost, grad] = cost_function(X, y, theta)
        theta = theta - alpha*grad
    return theta


def predict(X, theta):
    h = sigmoid(np.sum(theta*X, 1)) + 0.5
    return h.astype(int)


def test():
    from sklearn import datasets
    iris = datasets.load_iris()
    X_ = iris.data[0:100, :]
    X = np.c_[np.ones(100), X_]
    y = iris.target[0:100]
    theta = grad_descent(X, y, inter=35)
    h = predict(X, theta)
    print error_rate(y, h)


def error_rate(y, h):
    n = len(y)
    error = 0
    for i in range(n):
        if y[i] != h[i]:
            error += 1
    return 1.0*error/n


if __name__ == "__main__":
    test()
