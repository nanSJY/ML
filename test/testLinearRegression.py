from model import LinearRegression
from sklearn import linear_model,datasets

reg = LinearRegression()
reg1 = linear_model.LinearRegression()

boston = datasets.load_boston()
X = boston.data[0:100, 0].reshape(-1, 1)
y = boston.target[0:100].reshape(-1,1)
print reg.fit(X,y,n_inter=1000,alpha=0.01)
reg1.fit(X,y)
print reg1.intercept_,reg1.coef_