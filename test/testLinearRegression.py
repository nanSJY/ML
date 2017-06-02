from model import LinearRegression
from sklearn import linear_model,datasets
import matplotlib.pyplot as plt
import numpy as np


boston = datasets.load_boston()
X = boston.data[0:100, 12].reshape(-1, 1)
y = boston.target[0:100].reshape(-1, 1)
# X = np.array([[1],[2]])
# y = np.array([[2],[3]])

reg = LinearRegression()
reg.fit(X, y, n_inter=2000, alpha=1e-4)
print reg.intercept_, reg.coef_

reg1 = linear_model.LinearRegression()
reg1.fit(X, y)
print reg1.intercept_, reg1.coef_

plt.scatter(X, y)
x = np.arange(min(X), max(X), 0.1)
y = x*reg.coef_ + reg.intercept_
y1 = x*reg1.coef_[:][0] + reg1.intercept_
plt.plot(x, y1, x, y)
plt.show()
