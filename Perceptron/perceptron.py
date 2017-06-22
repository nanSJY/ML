import numpy as np


class Perceptron:

    def __init__(self, n_inter=10):
        self.coef_ = None
        self.intercept_ = None
        self.n_inter = n_inter

    def fit(self, X, y, coef_init=None, intercept_init=None, alpha=1):

        [m, n] = X.shape
        if y.shape[0] != m:
            raise ValueError('X.shape[1] should equal to y.shape')

        if coef_init is None:
            coef_init = np.zeros([n])
        if intercept_init is None:
            intercept_init = np.zeros([1])

        self.coef_ = coef_init
        self.intercept_ = intercept_init
        for j in range(self.n_inter):
            for i in range(m):
                if (self.coef_.dot(X[i, :]) + self.intercept_)*y[i] <= 0:
                    self.coef_ += alpha*y[i]*X[i, :]
                    self.intercept_ += alpha*y[i]
                    break
        return
