# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 18:51:01 2021

@author: nishant
"""
import pandas as pd
import config
import os
from glob import glob
from functools import reduce

df = []
for name in glob(os.path.join(config.ENSEMBLE,'*.csv')):
      df.append(pd.read_csv(name))

df = reduce(lambda  left,right: pd.merge(left,right,on=['customer_id'],how='outer'), df)
pred = df.drop(columns=['customer_id']).mode(axis=1)
submission_ensemble = pd.DataFrame({'customer_id':df['customer_id'],'credit_card_default':pred[0]})
submission_ensemble.to_csv(os.path.join(config.SUBMIT,'submission_ensemble.csv'),index=False)   