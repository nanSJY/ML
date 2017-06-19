# -*- coding: UTF-8 -*-
"""
Created on 2017-06-19

@author: zzx
"""

from KdTree import KdTree
import numpy as np


def predict(X_train, y_train, X_test):
    t = KdTree(X_train.shape[0])
    root = t.create_tree(X_train)
    nearest_data = []
    for x in X_test:
        nearest_data.append(t.find_nn(root, x).tolist())

    X_train_list = X_train.tolist()
    index = []
    for n in nearest_data:
        index.append(X_train_list.index(n))

    return y_train[index]


def test():
    from sklearn import datasets
    import random

    iris = datasets.load_iris()
    X_ = iris.data[0:100, :]
    X = np.c_[np.ones(100), X_]
    y = iris.target[0:100]

    index = random.sample(range(100), 70)
    index_ = []
    for i in range(100):
        if i not in index:
            index_.append(i)

    X_train = X[index, :]
    y_train = y[index]
    X_test = X[index_, :]
    y_test = y[index_]
    print predict(X_train, y_train, X_test)
    print y_test

if __name__ == '__main__':
    test()