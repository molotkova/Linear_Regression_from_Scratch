import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


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


data = pd.read_csv("data_stage3.csv")
X = data.iloc[:, 0:2]
y = data.iloc[:, -1]
#X["Capacity"] = 1 / X["Capacity"]

lin_reg = CustomLinearRegression(fit_intercept=True)
lin_reg.fit(X, y)
y_hat = lin_reg.predict(X)
r2 = lin_reg.r2_score(y, y_hat)
rmse = lin_reg.rmse(y, y_hat)
results = {'Intercept': lin_reg.intercept, 'Coefficient': lin_reg.coefficient, 'R2': r2, 'RMSE': rmse}
print(results)
