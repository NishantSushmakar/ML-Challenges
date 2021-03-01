# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:03:46 2021

@author: nishant
"""
import pandas as pd
import numpy as np
import config
import os
from sklearn.preprocessing import PolynomialFeatures


def feature_generate():
    '''
    Function Generates features for the DataFrame

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    new dataframe

    '''
    df =pd.read_csv(config.TRAIN_PATH)
    
    pf = PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
    pf.fit(df.loc[:,['Upper_Age','Lower_Age','Reco_Policy_Premium']])
    poly_feat = pf.transform(df.loc[:,['Upper_Age','Lower_Age','Reco_Policy_Premium']])
    
    df['sum_low_up_reco'] = df['Upper_Age'] + df['Lower_Age'] + df['Reco_Policy_Premium']
    df['mean_low_up']  = (df['Upper_Age'] + df['Lower_Age'])/2.0
    
    df['city_region'] = df['City_Code'] +'_'+df['Region_Code'].astype(str)
    df['Lower_Age_cat'] = pd.cut(df['Lower_Age'],bins=5,labels=False)
    df['Upper_Age_cat'] = pd.cut(df['Upper_Age'],bins=5,labels=False)
    df['Reco_Policy_Premium'] = pd.cut(df['Lower_Age'],bins=10,labels=False)
    
    df['dur_type'] = df['Holding_Policy_Duration'].astype(str) +'_'+df['Holding_Policy_Type'].astype(str)
    
    df_poly_feat = pd.DataFrame(poly_feat,columns=pf.get_feature_names())
    df = pd.concat([df,df_poly_feat],axis=1)
    
    df.drop(columns=['Upper_Age','Lower_Age','Reco_Policy_Premium'],axis=1)
    df.to_csv(os.path.join(config.DATA_PATH,'cleaned_feat.csv'),index=False)
    
feature_generate()

    
    