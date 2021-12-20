# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:55:30 2021

@author: nishant
"""
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import os
import config
import category_encoders as ce


def ohe_feature(train_df,test_df,cat_col):
    
    
    ohe = OneHotEncoder()
    
    train = ohe.fit_transform(train_df[cat_col]).toarray()
    test = ohe.transform(test_df[cat_col]).toarray()
    
    columns = ohe.get_feature_names(cat_col)
    train_cat=pd.DataFrame(train,columns=columns)
    test_cat =pd.DataFrame(test,columns=columns)
    
    return train_cat,test_cat


def le_feature(train_df,test_df,cat_col):
    
    
    for col in cat_col:
        le =  LabelEncoder()
        le.fit(train_df[col])
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        
        
    return train_df,test_df
def catencoder(train_df,test_df):
    
    X = train_df.drop(columns=['customer_id','name','credit_card_default'])
    col = X.columns
    target = train_df.credit_card_default
    
    cbe_encoder = ce.cat_boost.CatBoostEncoder()
    cbe_encoder.fit(X,target)
    train_df[col] = cbe_encoder.transform(X)
    test_df[col] = cbe_encoder.transform(test_df[col])
    return train_df,test_df
    
def woe_encoder(train_df,test_df):
    col = ['age','net_yearly_income','no_of_days_employed','yearly_debt_payments',
             'credit_limit','credit_limit_used(%)','credit_score',
           'prev_defaults','total_family_members','no_of_children','gender',
           'owns_car','owns_house','occupation_type','migrant_worker','default_in_last_6months']
    
    num_feat = ['age','net_yearly_income','no_of_days_employed','yearly_debt_payments',
                           'credit_limit','credit_limit_used(%)','credit_score','prev_defaults',
                           'total_family_members','no_of_children']
    
    
    for c in num_feat:
        print(c,train_df[c].quantile(0.25),train_df[c].quantile(0.75))
        train_df[c] = pd.qcut(train_df[c],q=4,duplicates='drop')
        test_df[c] = pd.qcut(test_df[c],q=4,duplicates='drop')
    
    woe_encoder = ce.WOEEncoder(cols=col)
    woe_encoded_train = woe_encoder.fit_transform(train_df[col], train_df['credit_card_default']).add_suffix('_woe')
    woe_encoded_test = woe_encoder.transform(test_df[col]).add_suffix('_woe')
    
    
    train_df = pd.concat([woe_encoded_train,train_df[['credit_card_default','name','customer_id']]],axis=1)
    test_df = pd.concat([test_df[['customer_id','name']],woe_encoded_test],axis=1)
    
    return train_df,test_df
    
    
    
def preprocess(train_df,test_df,encoding_type):
    
    cat_col = ['gender', 'owns_car', 'owns_house', 'occupation_type','migrant_worker','default_in_last_6months']
    if encoding_type == 'ohe':
        train_cat,test_cat = ohe_feature(train_df, test_df, cat_col)
        
        train_df = train_df.drop(columns=cat_col)
        test_df = test_df.drop(columns=cat_col)
        
        train_df = pd.concat([train_df,train_cat],axis=1)
        test_df = pd.concat([test_df,test_cat],axis=1)
    elif encoding_type=='le':
        train_df,test_df = le_feature(train_df,test_df,cat_col)
    elif encoding_type =='cat':
        train_df,test_df = catencoder(train_df,test_df)
    else:
        train_df,test_df = woe_encoder(train_df,test_df)
    
    return train_df,test_df


if __name__ == '__main__':
    
    train_df = pd.read_csv(config.TRAIN)
    test_df = pd.read_csv(config.TEST)
    encoding_type = 'woe'
    train_df,test_df = preprocess(train_df,test_df,encoding_type)
    train_df.to_csv(os.path.join(config.LOC,f'train_{encoding_type}.csv'),index=False)
    test_df.to_csv(os.path.join(config.LOC,f'test_{encoding_type}.csv'),index=False)

    