from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
import model

iris_bunch = datasets.load_iris()
X = iris_bunch.data[0:100, 1:3].reshape(100, 2)
y = iris_bunch.target[0:100]
y[0:50] = -1

# clf = linear_model.Perceptron()
clf = model.Perceptron(n_inter=5)
clf.fit(X, y)

print('coefficient = ', clf.coef_)
print('intercept = ', clf.intercept_)
# print('inter:',clf.intercept_inter)

plt.scatter(X[0:50, 0], X[0:50, 1], c='r')
plt.scatter(X[50:100, 0], X[50:100, 1], c='y')
# clf.coef_ = clf.coef_[0]
k = -1.*clf.coef_[0]/clf.coef_[1]
b = -1.*clf.intercept_[0]/clf.coef_[1]
x = np.arange(np.min(X[:, 0]), np.max(X[:, 1]), 0.1)
y = k*x + b
plt.plot(x, y)
plt.show()