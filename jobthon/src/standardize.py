# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:38:27 2021

@author: nishant
"""
from sklearn.preprocessing import MinMaxScaler
import config
import pandas as pd
import os
import numpy as np

numerical_col = ['Region_Code','Upper_Age','Lower_Age','Reco_Policy_Premium']

df_train = pd.read_csv(config.TRAIN_PATH_OHE)
df_test =pd.read_csv(config.TEST_PATH)

for col in numerical_col : 
    
    mm_scaler = MinMaxScaler()
    df_train[col] = mm_scaler.fit_transform(np.array(df_train[col]).reshape((-1,1)))
    df_test[col] = mm_scaler.transform(np.array(df_test[col]).reshape((-1,1)))
    
    
df_train.to_csv(os.path.join(config.DATA_PATH,'train_norm_ohe.csv'),index=False)
df_test.to_csv(os.path.join(config.DATA_PATH,'test_norm_ohe.csv'),index=False)
