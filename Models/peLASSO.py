from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
from Model import Model
import numpy as np

class peLASSO(Model):

    def __init__(self,lambdas):
        self.name='peLASSO'
        self.lambda1 =lambdas[0]
        self.lambda2 = lambdas[1]
        self.indiv_mods={}

        super().__init__(self.name)

    def fit(self,X,y):
        X_preds=pd.DataFrame()
        for c in X.columns:
            m = linear_model.LinearRegression()
            m.fit(pd.DataFrame(X[c]),y)
            self.indiv_mods[c]=m
            pred=m.predict(pd.DataFrame(X[c]))
            X_preds[c]=pred
        self.mean=np.mean(y)
        self.selection_model=linear_model.Lasso(self.lambda1)
        self.selection_model.fit(X_preds,y)
        self.selection=self.selection_model.coef_!=0
        y_dem=y.array-X_preds.iloc[:,self.selection].mean(axis=1).transpose().array
        self.model=linear_model.Lasso(self.lambda2)
        if len(X_preds.iloc[:,self.selection].columns)>0:
            self.model.fit(X_preds.iloc[:,self.selection],y_dem)

    def predict(self,X):
        X_preds=pd.DataFrame()
        for c in X.columns:
            pred=self.indiv_mods[c].predict(pd.DataFrame(X[c]))
            X_preds[c]=pred
        if len(X_preds.iloc[:,self.selection].columns)<1:
            return self.mean * np.ones(len(X))
        return self.model.predict(X_preds.iloc[:,self.selection])+X_preds.iloc[:,self.selection].mean(axis=1).transpose().array






