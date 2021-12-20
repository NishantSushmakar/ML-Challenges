# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:58:17 2021

@author: nishant
"""
import joblib
import config
import pandas as pd
import os
model = 'cat'
for i in range(10):
    fold = f'{i}'
    df = pd.read_csv(config.TEST)
    test = df.drop(['customer_id','name'],axis=1)   
    clf = joblib.load(os.path.join(config.MODEL_DIR,f'{model}_{fold}.bin'))
    pred = clf.predict(test)
    submission = pd.DataFrame({'customer_id':df.customer_id,'credit_card_default':pred})
    submission.to_csv(os.path.join(config.SUBMIT,f'submission_{model}_{i}.csv'),index=False)