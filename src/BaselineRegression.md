---
title: BaselineRegression.py
---
``` python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from scipy.sparse import lil_matrix
from sklearn.linear_model import Ridge

class MeanRegression(BaseEstimator):
    def fit(self,X,y):
        self.mu = np.mean(y)
        df_train = pd.DataFrame({"bid":X[:,0],"uid":X[:,1],"rating":y.reshape(-1)})
        self.b_dict = (df_train.groupby("bid").mean().rating-self.mu).to_dict()
        self.u_dict = (df_train.groupby("uid").mean().rating-self.mu).to_dict()
    def predict(self,X):
        X = np.array(X,dtype=int).reshape(-1,2)
        result = []
        for row in X:
            b_bias = self.b_dict[row[0]] if row[0] in self.b_dict else 0
            u_bias = self.u_dict[row[1]] if row[1] in self.u_dict else 0
            result.append(self.mu+b_bias+u_bias)
        return np.array(result).reshape(-1)

    def score(self,X,y):
        return r2_score(y,self.predict(X))

    def mse(self,X,y):
        return np.mean((y-self.predict(X))**2)

class SparseRidgeRegression(BaseEstimator):
    def __init__(self, alpha=0, max_iter=1500, tol=1e-5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self,X,y):
        X = np.array(X,dtype=int).reshape(-1,2)
        y = np.array(y).reshape(-1,1)

        bids = np.unique(X[:,0])
        uids = np.unique(X[:,1])
        self.b_dict = {bid:pos for pos,bid in enumerate(bids)}
        self.u_dict = {uid:pos for pos,uid in enumerate(uids)}
        bids_len = len(bids)
        design_matrix = lil_matrix((len(X),len(bids)+len(uids)))
        for rid,row in enumerate(X):
            design_matrix[rid,self.b_dict[row[0]]] = 1
            design_matrix[rid,bids_len+self.u_dict[row[1]]] = 1

        design_matrix = design_matrix.tocsr()
        ridge = Ridge(alpha=self.alpha, fit_intercept=True, copy_X=False,
                      solver='sag', max_iter=self.max_iter, tol=self.tol)
        ridge.fit(design_matrix,y)
        self.mu = ridge.intercept_
        coef = ridge.coef_.reshape(-1)
        for i in range(len(bids)):
            self.b_dict[bids[i]] = coef[i]
        for i in range(len(uids)):
            self.u_dict[uids[i]] = coef[bids_len+i]

        return self

    def predict(self,X):
        X = np.array(X,dtype=int).reshape(-1,2)
        result = []
        for row in X:
            b_bias = self.b_dict[row[0]] if row[0] in self.b_dict else 0
            u_bias = self.u_dict[row[1]] if row[1] in self.u_dict else 0
            result.append(self.mu+b_bias+u_bias)
        return np.array(result).reshape(-1)

    def score(self,X,y):
        return r2_score(y,self.predict(X))

    def mse(self,X,y):
        return np.mean((y-self.predict(X))**2)
```
