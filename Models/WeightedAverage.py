from sklearn import linear_model
import pandas as pd
from Model import Model

class WeightedAverage(Model):

    def __init__(self):
        self.name='WeightedAverage'
        self.indiv_mods={}
        super().__init__(self.name)

    def fit(self,X,y):
        for c in X.columns:
            model = linear_model.LinearRegression()
            model.fit(pd.DataFrame(X[c]),y)
            self.indiv_mods[c]=model

    def predict(self,X):
        preds=pd.DataFrame()
        for c in X.columns:
            preds[c]=self.indiv_mods[c].predict(pd.DataFrame(X[c]))
        return preds.mean(axis=1).to_numpy()

