# write your code here

import pandas as pd
import numpy as np


class CustomLinearRegression:

    def __init__(self, fit_intercept=True):

        self.fit_intercept = fit_intercept
        self.coefficient = []
        self.intercept = 0

    def fit(self, X, y):
        a = (len(X) * (X * y).sum() - X.sum() * y.sum()) / (len(X) * (X ** 2).sum() - X.sum() ** 2)
        b = (y.sum() - a * X.sum()) / len(X)
        self.coefficient.append(a)
        self.intercept = b
        self.coefficient = np.array(self.coefficient)


data = pd.DataFrame({'x': [4, 4.5, 5, 5.5, 6, 6.5, 7],
                     'y': [33, 42, 45, 51, 53, 61, 62]})

## example 1
#data = pd.DataFrame({'x': [4, 7],
#                     'y': [10, 16]})
## example 3
#data = pd.DataFrame({'x': [1, 4.5, 14, 3.8, 7, 19.4],
#                     'y': [106, 150.7, 200.9, 115.8, 177, 156]})
## example 2
#data = pd.DataFrame({'x': [1, 2, 3, 4, 5],
#                     'y': [0, 0, 0, 0, 0]})

X = data.iloc[:, 0]
y = data.iloc[:, 1]

lin_reg = CustomLinearRegression(fit_intercept=True)
lin_reg.fit(X, y)
result = {'Intercept': lin_reg.intercept, 'Coefficient': lin_reg.coefficient}
print(result)