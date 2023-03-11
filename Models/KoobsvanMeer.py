from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Model import Model
import itertools
import pandas as pd

class KoobsvanMeer(Model):

    def __init__(self,components):
        self.name='NonLinearFactor'
        self.components=components
        super().__init__(self.name)

    def fit(self,X,y):
        self.sc=StandardScaler()
        X = pd.DataFrame(self.sc.fit_transform(X), columns=X.columns)
        self.cross_effects = [i for i in itertools.combinations(X.columns, 2)]
        for i in X.columns:
            self.cross_effects.append([i,i])
        X_stand = X.join(pd.DataFrame({
            f'{x[0]} {x[1]}': X[x[0]] * X[x[1]] for x in self.cross_effects
        }))
        self.pc=PCA(n_components=self.components)
        self.pc.fit(X_stand)
        X_pc=self.pc.transform(X_stand)
        self.model=linear_model.LinearRegression()
        self.model.fit(X_pc,y)

    def predict(self,X):
        X = pd.DataFrame(self.sc.transform(X), columns=X.columns)
        self.cross_effects = [i for i in itertools.combinations(X.columns, 2)]
        for i in X.columns:
            self.cross_effects.append([i,i])
        X_cross = X.join(pd.DataFrame({
            f'{x[0]} {x[1]}': X[x[0]] * X[x[1]] for x in self.cross_effects
        }))
        X_pc=self.pc.transform(X_cross)
        return self.model.predict(X_pc)





