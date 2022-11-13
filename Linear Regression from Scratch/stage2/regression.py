import pandas as pd
import numpy as np


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
        return X @ self.coefficient


data = pd.DataFrame({'x': [4, 4.5, 5, 5.5, 6, 6.5, 7],
                     'w': [1, -3, 2, 5, 0, 3, 6],
                     'z': [11, 15, 12, 9, 18, 13, 16],
                     'y': [33, 42, 45, 51, 53, 61, 62]})

#data = pd.DataFrame({'x': [11, 2, 3, 4, 5],
#                     'w': [6, 90, 8, 9, 10],
#                     'z': [11, 12, 13, 14, 15],
#                     'y': [0, 0, 0, 0, 0]})

X = data.iloc[:, 0:3]
y = data.iloc[:, -1]

lin_reg = CustomLinearRegression(fit_intercept=False)
lin_reg.fit(X, y)
result = np.array(lin_reg.predict(X))
print(result)
