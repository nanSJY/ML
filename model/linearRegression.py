import numpy as np


class LinearRegression:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def cost(self, X, y, w):
        return np.sum((X*w - y)**2)/2

    def fit(self, X, y, alpha=0.01, n_inter=100, coef_init=None, intercept_init=None):
        [m, n] = X.shape
        if y.shape[0] != m:
            raise ValueError('X.shape[1] should equal to y.shape')

        if coef_init is None:
            coef_init = np.zeros([n])
        if intercept_init is None:
            intercept_init = np.zeros([1])

        ones = np.ones([m, 1])
        X = np.column_stack([ones, X])
        theta = np.row_stack([intercept_init, coef_init])

        for inter in range(n_inter):
            error_item = np.zeros(shape=(m, 1))
            for i in range(m):
                error_item[i] = (y[i] - X[i, :].dot(theta))
            for j in range(n+1):
                grad_j = 0
                for i in range(m):
                    grad_j += error_item[i]*X[i, j]
                theta[j] += grad_j*alpha

        # normal equation (m > n)
        # theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        self.intercept_ = theta[0]
        self.coef_ = theta[1]
        return


