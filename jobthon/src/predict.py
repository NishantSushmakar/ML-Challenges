# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:11:39 2021

@author: nishant
"""
import joblib
import config
import pandas as pd
import os

df = pd.read_csv(config.TEST_PATH_OHE)
test = df.drop('ID',axis=1)
clf = joblib.load(config.MODEL_SELECT)
pred = clf.predict(test)
submission = pd.DataFrame(pred,columns=['Response'])
submission = pd.concat([df['ID'],submission],axis=1)
submission.to_csv(os.path.join(config.SUBMIT,'submission.csv'),index=False)



