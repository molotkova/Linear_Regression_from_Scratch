import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston


class CustomLinearRegression:

    def __init__(self, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficient = []
        self.intercept = 0

    def fit(self, X, y):
        if self.fit_intercept:
            ones = [1] * X.shape[0]
            X = np.hstack((np.reshape(ones, (X.shape[0], 1)), X))
        #    a = (len(X) * (X * y).sum() - X.sum() * y.sum()) / (len(X) * (X ** 2).sum() - X.sum() ** 2)
        #    b = (y.sum() - a * X.sum()) / len(X)
        #    self.intercept = b
        #    self.coefficient = np.array([a])
        #else:
        #    b = np.linalg.inv(X.T @ X) @ X.T @ y
        #    self.coefficient = np.array(b)
        b = np.linalg.inv(X.T @ X) @ X.T @ y
        self.coefficient = np.array(b)

    def predict(self, X):
        if self.fit_intercept:
            ones = [1] * X.shape[0]
            X = np.hstack((np.reshape(ones, (X.shape[0], 1)), X))
            y_hat = X @ self.coefficient
            self.intercept = self.coefficient[0]
            self.coefficient = self.coefficient[1:]
        else:
            y_hat = X @ self.coefficient
        return y_hat

    def r2_score(self, y, y_hat):
        return 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def rmse(self, y, y_hat):
        return (((y - y_hat) ** 2).sum() / len(y)) ** 0.5

# data old version
#data = pd.read_csv("data.csv")
#X = data.iloc[:, :2]
#y = data.iloc[:, -1]
#X["Capacity"] = 1 / X["Capacity"]

# generating data
#data = pd.DataFrame(load_boston()['data'])
#data['target'] = load_boston()['target']
#X, y = data.iloc[:11, [2, 6, 10]], data.loc[:10, 'target']

# new data
data = pd.read_csv("data_stage4.csv")
X, y = data.drop("y", axis=1), data["y"]



lin_reg = CustomLinearRegression(fit_intercept=True)
lin_reg.fit(X, y)
y_hat = lin_reg.predict(X)
r2 = lin_reg.r2_score(y, y_hat)
rmse = lin_reg.rmse(y, y_hat)
results_custom = {'Intercept': lin_reg.intercept, 'Coefficient': lin_reg.coefficient, 'R2': r2, 'RMSE': rmse}

regSci = LinearRegression(fit_intercept=True)
regSci.fit(X, y)
y_hat = regSci.predict(X)
r2 = r2_score(y, y_hat)
rmse = mean_squared_error(y, y_hat) ** 0.5
results_sklearn = {'Intercept': regSci.intercept_, 'Coefficient': regSci.coef_, 'R2': r2, 'RMSE': rmse}

difference = {}
for i, j in zip(results_custom.items(), results_sklearn.items()):
    difference[i[0]] = abs(i[1] - j[1])

print(difference)
