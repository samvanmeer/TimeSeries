from sklearn import linear_model
import pandas as pd
from Model import Model
class AR1(Model):

    def __init__(self):
        self.name='AR(1)'
        super().__init__(self.name)

    def fit(self,X,y):
        self.model=linear_model.LinearRegression()
        self.model.fit(pd.DataFrame(X['EMPL']),y)

    def predict(self,X):
        return self.model.predict(pd.DataFrame(X['EMPL']))





