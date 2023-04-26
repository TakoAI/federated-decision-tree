# Federated Linear Regression

# Dependencies
import pandas as pd
import numpy as np
from sklearn import linear_model

class FLR:
    """
    Class for Federated Linear Regression
    """
    def __init__(
        self,
        values: (np.array, float, float, float, int) = None # coef, intercept, min, max, count
    ):
        self.values = values
    
    def fit(
        self,
        X: pd.DataFrame,
        Y: list
    ):
        reg = linear_model.LinearRegression().fit(X, Y)
        self.values = (reg.coef_, reg.intercept_, min(Y), max(Y), len(Y))
        return self
        
    def predict(self, X):
        coef, intercept, minv, maxv, _ = self.values
        return np.clip(np.dot(X, coef) + intercept, minv, maxv)
    
    def predict_one(self, X1):
        coef, intercept, minv, maxv, _ = self.values
        return min(max(np.sum(X1 * coef) + intercept, minv), maxv)
    
    @staticmethod
    def weightedAvg(v1, c1, v2, c2):
        return (v1 * c1 + v2 * c2) / (c1 + c2)

    def merge(self, lr):
        if lr:
            self.values = (
                self.weightedAvg(self.values[0], self.values[4], lr.values[0], lr.values[4]),
                self.weightedAvg(self.values[1], self.values[4], lr.values[1], lr.values[4]),
                min(self.values[2], lr.values[2]),
                max(self.values[3], lr.values[3]),
                self.values[4] + lr.values[4]
            )

    def copy(self):
        values = (np.copy(self.values[0]), self.values[1], self.values[2], self.values[3], self.values[4])
        return FLR(values)
