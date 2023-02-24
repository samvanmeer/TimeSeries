from sklearn import linear_model
import numpy as np
from Model import Model

class HistoricalAverage(Model):

    def __init__(self):
        self.name='HistoricalAverage'
        super().__init__(self.name)

    def fit(self,X,y):
        self.mean=np.mean(y)

    def predict(self,X):
        return self.mean*np.ones(len(X))





