# Federated Bayes Decision

# Dependencies
import pandas as pd
import numpy as np
import math

class FBD:
    """
    Class for Federated Bayes Decision
    """
    def __init__(
        self,
        values: dict[int, (np.array, np.array, int)] = {} # mean, std, count
    ):
        self.values = values
    
    def fit(
        self,
        X: pd.DataFrame,
        Y: list
    ):
        self.values = {}
        yarr = np.array(Y)
        ks = np.unique(yarr)
        df = X.copy()
        for k in ks:
            ymatch = yarr == k
            self.values[k] = [
                np.array(np.mean(df[ymatch], axis=0)),
                np.array(np.std(df[ymatch], axis=0)) + 1e-150,
                sum(ymatch)
            ]
        return self

    @staticmethod
    def probability(X, vmean, vstd):
        return np.exp(-np.sum(((X - vmean) / vstd) ** 2, axis=1)) / np.sqrt(sum(vstd ** 2) * 2 * np.pi)
        
    def predict(self, X):
        npx = X.to_numpy()
        lx = len(X)
        kp = {
            k: self.values[k][2] * self.probability(npx, self.values[k][0], self.values[k][1]) for k in self.values
        }
        if lx == 1:
            return max(kp, key=kp.get)
        yp = []
        for i in range(lx):
            hk = 0
            hv = 0
            for k in kp:
                try:
                    if kp[k][i] > hv:
                        hk = k
                        hv = kp[k][i]
                except:
                    print(kp, k, i)
                    asdffds
            yp.append(hk)
        return yp
    
    @staticmethod
    def probability_one(X1, vmean, vstd):
        return np.exp(-np.sum(((X1 - vmean) / vstd) ** 2)) / np.sqrt(sum(vstd ** 2) * 2 * np.pi)
    
    def predict_one(self, X1):
        return {
            k: self.values[k][2] * self.probability_one(X1, self.values[k][0], self.values[k][1]) for k in self.values
        }        
    
    @staticmethod
    def weightedAvg(v1, c1, v2, c2):
        return (v1 * c1 + v2 * c2) / (c1 + c2)

    def merge(self, bd):
        if bd:
            for k in self.values:
                if k in bd.values:
                    nmean = self.weightedAvg(self.values[k][0], self.values[k][2], bd.values[k][0], bd.values[k][2])
                    nstdself = np.sqrt(((self.values[k][0] - nmean) ** 2) + self.values[k][1] ** 2)
                    nstdbd = np.sqrt(((bd.values[k][0] - nmean) ** 2) + bd.values[k][1] ** 2)
                    self.values[k] = (
                        nmean,
                        self.weightedAvg(nstdself, self.values[k][2], nstdbd, bd.values[k][2]),
                        self.values[k][2] + bd.values[k][2]
                    )
            for k in bd.values:
                if k not in self.values:
                    self.values[k] = bd.values[k]

    def copy(self):
        values = {
            k: (np.copy(self.values[k][0]), np.copy(self.values[k][1]), self.values[k][2]) for k in self.values
        }
        return FBD(values)
