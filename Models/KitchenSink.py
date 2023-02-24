from sklearn import linear_model
from Model import Model

class KitchenSink(Model):
    def __init__(self):
        self.name='KitchenSink'
        super().__init__(self.name)

    def fit(self,X,y):
        self.model=linear_model.LinearRegression()
        self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)





