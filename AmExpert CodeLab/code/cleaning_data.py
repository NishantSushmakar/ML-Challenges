# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 13:28:17 2021

@author: nishant
"""
import pandas as pd
import config
import os
from sklearn.impute import SimpleImputer

def outlier_capping(train_df,test_df):
    num_feat = ['age','net_yearly_income','no_of_days_employed','yearly_debt_payments',
                           'credit_limit','credit_limit_used(%)','credit_score','prev_defaults',
                           'total_family_members','no_of_children']
    for col in num_feat:
        test_df[col] = test_df[col].clip(upper=train_df[col].quantile(0.99))
        test_df[col] = test_df[col].clip(lower=train_df[col].quantile(0.01))
        
        train_df[col] = train_df[col].clip(upper=train_df[col].quantile(0.99))
        train_df[col] = train_df[col].clip(lower=train_df[col].quantile(0.01))
    
    
    return train_df,test_df

def missing_values(train_df,test_df):
    
    cat_col = ['owns_car','migrant_worker']
    num_col = ['no_of_children','no_of_days_employed','total_family_members','yearly_debt_payments','credit_score']
    
    
    imp = SimpleImputer(strategy="most_frequent")
    train_df[cat_col] = imp.fit_transform(train_df[cat_col])
    test_df[cat_col] = imp.transform(test_df[cat_col])
    
    imp_num = SimpleImputer(strategy='median')
    train_df[num_col] = imp_num.fit_transform(train_df[num_col])
    test_df[num_col] = imp_num.transform(test_df[num_col])
    
    train_df,test_df =  outlier_capping(train_df,test_df)
    
    
    return train_df,test_df    



if __name__ == '__main__':
    
    train_df = pd.read_csv(config.TRAIN)
    test_df = pd.read_csv(config.TEST)
    
    train_df,test_df = missing_values(train_df,test_df)
    
    
    train_df.to_csv(os.path.join(config.LOC,'train_clean.csv'),index=False)
    test_df.to_csv(os.path.join(config.LOC,'test_clean.csv'),index=False)
    