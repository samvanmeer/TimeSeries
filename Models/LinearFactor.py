from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Model import Model

class LinearFactor(Model):

    def __init__(self,components):
        self.name='LinearFactor'
        self.components=components
        super().__init__(self.name)

    def fit(self,X,y):
        self.sc=StandardScaler()
        X_stand=self.sc.fit_transform(X)
        self.pc=PCA(n_components=self.components)
        self.pc.fit(X_stand)
        X_pc=self.pc.transform(X_stand)
        self.model=linear_model.LinearRegression()
        self.model.fit(X_pc,y)

    def predict(self,X):
        X_pc=self.pc.transform(self.sc.transform(X))
        return self.model.predict(X_pc)





