import itertools
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
from Model import Model
import numpy as np

class Lasso_interactions(Model):

    def __init__(self,lam):
        self.name='Lasso_interactions'
        self.lambda1 =lam

        super().__init__(self.name)

    def fit(self,X,y):
        self.sc=StandardScaler()
        X = pd.DataFrame(self.sc.fit_transform(X), columns=X.columns)
        self.cross_effects = [i for i in itertools.combinations(X.columns, 2)]
        for i in X.columns:
            self.cross_effects.append([i,i])
        X_cross = X.join(pd.DataFrame({
            f'{x[0]} {x[1]}': X[x[0]] * X[x[1]] for x in self.cross_effects
        }))
        self.model=linear_model.Lasso(self.lambda1)
        self.model.fit(X_cross,y)

    def predict(self,X):
        X = pd.DataFrame(self.sc.transform(X), columns=X.columns)
        self.cross_effects = [i for i in itertools.combinations(X.columns, 2)]
        for i in X.columns:
            self.cross_effects.append([i,i])
        X_cross = X.join(pd.DataFrame({
            f'{x[0]} {x[1]}': X[x[0]] * X[x[1]] for x in self.cross_effects
        }))
        return self.model.predict(X_cross)








def fit(self, X, y, normalization, fold_interactions):
    X = pd.DataFrame(self.dataset.normalize(X, normalization), columns=X.columns)
    self.cross_effects = fold_interactions
    X_cross = X.join(pd.DataFrame({
        f'{x} {y}': X.iloc[:, x] * X.iloc[:, y] for x, y in self.cross_effects
    }))
    self.model = Logistic_Regression.LogReg(self.dataset, penalty=self.penalty, C=self.C)
    self.model.fit(X_cross, y, 'none')
    return self.model


def predict(self, X, normalization):
    X = pd.DataFrame(self.dataset.normalize(X, normalization), columns=X.columns)
    X_cross = X.join(pd.DataFrame({
        f'{x} {y}': X.iloc[:, x] * X.iloc[:, y] for x, y in self.cross_effects
    }))
    return self.model.predict(X_cross, 'none')