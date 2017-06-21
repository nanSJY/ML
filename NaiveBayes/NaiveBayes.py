# -*- coding: UTF-8 -*-
"""
Created on 2017-06-21

@author: zzx
"""

import numpy as np


class MultinomialNB(object):
    """
    classes: 类别 y 的所有可能取值
    K_classes： label 的个数

    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.classes = None
        self.class_prior_prob = None
        self.conditional_prob = None

    def fit(self, X, y):
        self.calculate_class_prior_prob(y)
        self.calculate_conditional_prob(X, y)

    # 计算 Y 的先验概率
    def calculate_class_prior_prob(self, y):
        self.classes = np.unique(y)
        K_classes = len(self.classes)
        m = len(y)
        self.class_prior_prob = []
        for c in self.classes:
            c_num = np.sum(np.equal(y, c))
            self.class_prior_prob.append((c_num + self.alpha)*1.0/(m + K_classes*self.alpha))
        return self.class_prior_prob

    def calculate_conditional_prob(self, X, y):
        # {c1: {x0:{value0: 0.2, value1: 0.8}, x1:{},  }, c2:{}, }
        self.conditional_prob = {}
        # [m, n] = X.shape
        n = len(X[0])
        for c in self.classes:
            self.conditional_prob[c] = {}
            for i in range(n):
                feature = X[np.equal(y, c)][:, i]
                self.conditional_prob[c][i] = self.calculate_feature_prob(feature)
        return self.conditional_prob

    def calculate_feature_prob(self, feature):
        # s_i 第i个特征的可能取值
        # num 第i个特征取值个数
        s_i = np.unique(feature)
        num = len(s_i)
        c = len(feature)
        prob = {}
        for s in s_i:
            prob[s] = (np.sum(np.equal(feature, s)) + self.alpha)*1.0/(c + num*self.alpha)
        return prob

    def predict(self, x):
        prob = []
        for k in range(len(self.classes)):
            pi = 1
            for j in range(2):
                pi *= self.conditional_prob[self.classes[k]][j][x[j]]
            prob.append(self.class_prior_prob[k]*pi)
        return self.classes[np.argmax(prob)]

if __name__ == '__main__':
    X = np.array([[1,1],[1,2],[1,2],[1,1],[1,1],
                  [2,1],[2,2],[2,2],[2,3],[2,3],
                  [3,3],[3,2],[3,2],[3,3],[3,3],])
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    nb = MultinomialNB(alpha=1)
    nb.fit(X, y)
    print nb.conditional_prob
    print nb.class_prior_prob
    print nb.predict(np.array([3,3]))








