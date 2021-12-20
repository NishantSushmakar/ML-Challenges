# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:47:11 2021

@author: nishant
"""
import pandas as pd
import numpy as np
import config
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def feature(train_df,df,num_feat):
    
    
    df['have_child']=df['no_of_children'].apply(lambda x : 0 if x==0 else 1)
    df['no_of_adults'] = df['total_family_members']-df['no_of_children']
    df['credit_used'] = (df['credit_limit_used(%)']/100)*df['credit_limit']
    
    df['is_a_defaulter'] = df['prev_defaults'].apply(lambda x: 0 if x==0 else 1)
    df['debt_by_used'] =df['yearly_debt_payments']/df['credit_used']
    df['debt_by_income'] = df['yearly_debt_payments']/df['net_yearly_income']
    
    
    for col in num_feat:
        df[f'{col}_cat'] = pd.cut(df[col],bins=4,include_lowest=True,labels=[0,1,2,3])
        df[f'{col}_log'] = df[col].apply(lambda x:np.log(1+x))
    
    num_col = ['age','net_yearly_income','no_of_days_employed','yearly_debt_payments',
                           'credit_limit','credit_limit_used(%)','credit_score','prev_defaults',
                           'total_family_members','no_of_children','no_of_adults','credit_used','debt_by_used'
                           ,'debt_by_income']
    
    df['sum'] = df[num_feat].sum(axis=1)
    df['mean'] = df[num_feat].mean(axis=1)
    df['median'] = df[num_feat].median(axis=1)
    df['min'] = df[num_feat].min(axis=1)
    df['max'] = df[num_feat].max(axis=1)
    df['std'] = df[num_feat].std(axis=1)
    
    
    
    
    return df


def cluster(train_df,test_df,num_feat):
    
    X_train = train_df[num_feat]
    X_valid = test_df[num_feat]
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_valid = scale.transform(X_valid)
    
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X_train)
    train_df['kmeans'] = kmeans.predict(X_train)
    test_df['kmeans'] = kmeans.predict(X_valid)
    return train_df,test_df



if __name__ =='__main__':
    
    train_df = pd.read_csv(config.TRAIN)
    test_df = pd.read_csv(config.TEST)
    
    num_feat = ['age','net_yearly_income','no_of_days_employed','yearly_debt_payments',
                           'credit_limit','credit_limit_used(%)','credit_score','prev_defaults',
                           'total_family_members','no_of_children']
    
    train_df = feature(train_df,train_df,num_feat)
    test_df = feature(train_df,test_df,num_feat)
    
    train_df,test_df = cluster(train_df,test_df,num_feat)
    
    train_df.to_csv(os.path.join(config.LOC,'train_feat_ohe.csv'),index=False)
    test_df.to_csv(os.path.join(config.LOC,'test_feat_ohe.csv'),index=False)
