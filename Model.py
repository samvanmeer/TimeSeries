import utils
import dateutil.relativedelta
import pandas as pd
import numpy as np
from feval import helpers  # to easily compute losses
from feval import cmcs

from sklearn import metrics
class Model:
    def __init__(self,name):
        self.name=name

    def moving_window_pred(self,X,y,start_string='1980-01-01',end_string='2022-09-01',years=40):
        start=utils.dformat(start_string)
        end=utils.dformat(end_string)
        i=start
        preds=[]

        while i<=end:
            end_training=i-dateutil.relativedelta.relativedelta(months=1)
            begin_training=end_training-dateutil.relativedelta.relativedelta(years=years)
            X_train=X[begin_training:end_training]
            y_train=y[begin_training:end_training]
            x_pred=X.loc[i,:]
            self.fit(X_train,y_train)
            preds.append(self.predict(pd.DataFrame(x_pred).transpose())[0])
            i=i+dateutil.relativedelta.relativedelta(months=1)
        preds_df=pd.DataFrame(preds)
        preds_df.index=pd.DatetimeIndex(X[start:end].index)
        mse=metrics.mean_squared_error(y[start:end],preds_df)
        return mse,preds_df,y[start:end]

def evaluation(preds,target):
    L = -1*helpers.se(target, preds)  # Squared loss

    # Perform the cmcs with an HAC estimator, the Bartlett kernel and a significance level of 0.01
    mcs, S, cval, pval, removed = cmcs(L, alpha=0.05, covar_style="hac", kernel="Bartlett")

    return mcs, S,cval,pval,removed

