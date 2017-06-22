# -*- coding: UTF-8 -*-
"""
Created on 2017-06-22

@author: zzx
"""

import numpy as np
from scipy.stats import mode


class DecisionTree:

    def __init__(self):
        self.root = None

    def fit(self, D, A, sigma):
        self.root = self.build_tree(D, A, sigma)

    def build_tree(self, D, A, sigma):
        if len(np.unique(D[:, -1])) == 1:
            return D[0, -1]
        if A is None:
            return mode(y,axis=None)[0][0]
        opt_feature, g, subset = self.optimal_feature(D, A)
        if g < sigma:
            return mode(y, axis=None)[0][0]
        else:
            A.remove(opt_feature)
            Tree = {opt_feature: {}}
            for key in subset:
                Tree[opt_feature][key] = self.build_tree(subset[key], A, sigma)
            return Tree


    # 训练集D={X, y}，特征集 feature_set
    def optimal_feature(self, D, feature_set):
        empirical_entropy = self.calc_empirical_entropy(D)

        # A 对 D 的经验条件熵
        # 特征 A 有 n 个取值 {a1, a2, ... , an}, 把 D 划分为 n 个子集 D1, D2, ... , Dn
        # 子集 Di 根据标签分为 Di1, ... , Dik
        opt_feature = feature_set[0]
        min_empirical_condition_entropy = None
        subset = None

        for A in feature_set:
            empirical_condition_entropy = 0
            A_values = np.unique(D[:, A])
            A_D = {}
            for a in A_values:
                A_D[a] = (D[np.equal(D[:, A], a)])
            for D_i in A_D.values():
                empirical_condition_entropy += float(len(D_i))/len(D)*self.calc_empirical_entropy(D_i)
            # print empirical_entropy- empirical_condition_entropy
            if (min_empirical_condition_entropy is None) or (empirical_condition_entropy < min_empirical_condition_entropy) :
                opt_feature = A
                min_empirical_condition_entropy = empirical_condition_entropy
                subset = A_D

        return opt_feature,empirical_entropy-min_empirical_condition_entropy,subset

    # D 的经验熵
    def calc_empirical_entropy(self, D):
        X = D[:, :-1]
        y = D[:, -1]
        classes = np.unique(y)
        c_k = np.zeros([len(classes)])  # D 中每个 LABEL 对应的样本数量
        for i in range(len(classes)):
            c_k[i] = (float(np.sum(np.equal(y, classes[i]))))
        return -np.sum(c_k/len(y) * np.log2(c_k/len(y)))

    def show(self):
        if self.root:
            import treePlotter
            treePlotter.createPlot(self.root)

    def predict(self, x):
        return self._classify(x, self.root)

    def _classify(self, x, root):
        if isinstance(root, int):
            return root
        feature = root.keys()[0]
        x_feature = x[feature]
        D = root[feature][x_feature]
        return self._classify(x, D)

if __name__ == '__main__':
    y = np.array([[0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]])
    X = np.array([[1, 0, 0, 0],
                  [1, 0, 0, 1],
                  [1, 1, 0, 1],
                  [1, 1, 1, 0],
                  [1, 0, 0, 0],
                  [2, 0, 0, 0],
                  [2, 0, 0, 1],
                  [2, 1, 1, 1],
                  [2, 0, 1, 2],
                  [2, 0, 1, 2],
                  [3, 0, 1, 2],
                  [3, 0, 1, 1],
                  [3, 1, 0, 1],
                  [3, 1, 0, 2],
                  [3, 0, 0, 0]])

    D = np.concatenate((X, y.T), axis=1)
    t = DecisionTree()
    t.fit(D, A=[0,1,2,3], sigma=0)
    # t.show()
    print t.root
    p = []
    for i in range(15):
        p.append(t.predict(X[i, :]))
    print 'p=',p
    print 'y=',y.tolist()[0]
