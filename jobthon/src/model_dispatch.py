# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:22:01 2021


@author: nishant
"""
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
models = {
       "random_forest_gini" : RandomForestClassifier(),
       "random_forest_entropy" : RandomForestClassifier(criterion='entropy'),
       "xgb" : XGBClassifier(n_estimators = 1000,max_depth = 5 ,use_label_encoder=False),
       'knn' : KNeighborsClassifier(n_neighbors=5),
       'lr' : LogisticRegression(),
       'svm' : SVC(kernel='rbf'),
       'gnb' : GaussianNB(),
       'mlp':MLPClassifier(max_iter=2000)
       
    }
