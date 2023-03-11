import dateutil
import pandas as pd

import utils
from Models import KitchenSink
from Models import LinearFactor
from Models import WeightedAverage
from Models import HistoricalAverage
from Models import AR1
from Models import peLASSO
from sklearn import metrics
import Model

#Read and structure data
from Models.KoobsvanMeer import KoobsvanMeer
from Models.Lasso_interactions import Lasso_interactions
from utils import dformat

df=pd.read_excel('USEMP.xlsx')
df=df.set_index('Month')
df.index = pd.to_datetime(df.index)
y=df['EMP']
X=df.drop('EMP',axis=1)


#Specify models
nr_components=2 #For PCA
lambdas=[0.02,0.2] #For peLASSO. These values are from Diebold(2019)
interaction_p=0.2
mod1=KitchenSink.KitchenSink()
mod2=WeightedAverage.WeightedAverage()
mod3=LinearFactor.LinearFactor(nr_components)
mod4=peLASSO.peLASSO(lambdas)
mod5=Lasso_interactions(interaction_p)
mod6=KoobsvanMeer(5)
bm1= HistoricalAverage.HistoricalAverage()
bm2= AR1.AR1()
m_list=[mod1,mod2,mod3,mod4,mod5,mod6,bm1,bm2]
mods={m.name: m for m in m_list}

#Fit and predict in sample
preds={}
mses={}
preds_os=pd.DataFrame()
mses_os={}
for mod in mods.values():
    mod.fit(X,y)
    preds[mod.name]=mod.predict(X)
    mses[mod.name]=metrics.mean_absolute_error(y,preds[mod.name])
    mse_os, pred_os,y_os=mod.moving_window_pred(X, y)
    mses_os[mod.name]=mse_os
    preds_os[mod.name]=pred_os

mcs=pd.DataFrame()
for i in range(0,len(preds_os[dformat('1990-01-01'):])):
    X_temp=preds_os[(dformat('1980-01-01')+dateutil.relativedelta.relativedelta(months=i)):(dformat('1980-01-01')+dateutil.relativedelta.relativedelta(months=i+120))].to_numpy()
    y_temp= y_os[(dformat('1980-01-01')+dateutil.relativedelta.relativedelta(months=i)):(dformat('1980-01-01')+dateutil.relativedelta.relativedelta(months=i+120))].to_numpy()
    res=Model.evaluation(X_temp,y_temp)
    mcs[dformat('1990-01-01')+dateutil.relativedelta.relativedelta(months=i)]=res[0][0].T

a=pd.DataFrame.from_dict(mses,orient='index')

mcs=mcs.T
scores=mcs.groupby(mcs.index.year).mean().T
scores.index=mods.keys()

utils.heatmap_plot(scores)
