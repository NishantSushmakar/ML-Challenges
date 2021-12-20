# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:26:45 2021

@author: nishant
"""
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
params_lgbm = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'lambda_l1': 5.412528875661013,
    'lambda_l2': 0.9505207663514971,
    'num_leaves': 41,
    'feature_fraction': 0.8218562341482682,
    'bagging_fraction': 0.5917453868603656,
    'bagging_freq': 6,
    'min_child_samples': 88
    }

models = {
       'rf' : RandomForestClassifier(),
       'xgb' : XGBClassifier(use_label_encoder=False),
       'lgbm':LGBMClassifier(**params_lgbm),
       'ridge':RidgeClassifier(),
       'et':ExtraTreesClassifier(),
       'lr':LogisticRegression(),
       'cat':CatBoostClassifier()
       
    }
