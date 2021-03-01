# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:41:06 2021

@author: nishant
"""
import numpy as np
from functools import partial 
from scipy.optimize import fmin
from sklearn import metrics

class optimizeAUC:
    
    def __init__(self):
        self.coef_ = 0
        
    def _auc(self,x,y,coef):
        
        x_coef = x*coef
        predictions = np.sum(x_coef,axis=1)
        auc_score = metrics.roc_auc_score(y,predictions)
        
        return -1.0*auc_score
    
    def fit(self,x,y):
        loss_partial = partial(self._auc,x,y)
        
        initial_coef = np.random.dirichlet(np.ones(x.shape[1]),size=1)
        print(initial_coef.shape)
        self.coef_ = fmin(loss_partial,initial_coef,disp=True)
        
    def predict(self,x):
        
        x_coef = x*self.coef_
        predictions = np.sum(x_coef,axis=1)
        return predictions
        
        