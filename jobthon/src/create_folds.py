# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:46:12 2021

@author: nishant
"""
import pandas as pd
import numpy as np
import config
import os
from sklearn import model_selection

def make_folds(df):
    '''
    Function create Stratified K folds on the dataset

    Parameters
    ----------
    df : dataframe

    Returns
    -------
    dataframe

    '''
    
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.Response.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_,'kfold'] = f
    
    return df


if __name__  == '__main__':
    
    df_ohe = pd.read_csv(config.TRAIN_PATH_OHE)
    df_le = pd.read_csv(config.TRAIN_PATH_LE)
    
    df_ohe = make_folds(df_ohe)
    df_le = make_folds(df_le)
    
    
    df_ohe.to_csv(os.path.join(config.DATA_PATH,'ohe_fold.csv'),index=False)
    df_le.to_csv(os.path.join(config.DATA_PATH,'le_fold.csv'),index=False)
    
    
    