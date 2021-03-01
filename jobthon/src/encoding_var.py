# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:44:07 2021

@author: nishant
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import bisect
import config



def ohe_cat_col(df_train,df_test,cat_col):
    '''
    One Hot Encode all the categorical columns

    Parameters
    ----------
    df_train : Training DataFrame
    df_test : Test DataFrame
    cat_col : Categorical Columns

    Returns
    -------
    df_train : Training DataFrame
    df_test : Test DataFrame

    '''
    
    col_for_le = ['city_region' ,'City_Code','dur_type']
    for col in col_for_le:
        le = LabelEncoder()
        le.fit(df_train[col].astype(str))
        df_test[col] = df_test[col].map(lambda s: 'other' if s not in le.classes_ else s)
        le_classes = le.classes_.tolist()
        bisect.insort_left(le_classes, 'other')
        le.classes_ = np.array(le_classes)
        df_train[col]  = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
                
    
    
    print('One Hot Encoding')
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df_train[cat_col].astype(str))
    arr = enc.transform(df_train[cat_col].astype(str)).toarray()
    df_train_ohe = pd.DataFrame(arr,columns=enc.get_feature_names())
    df_train = df_train.drop(columns=cat_col)
    df_train =pd.concat([df_train,df_train_ohe],axis=1)
    
    
    test_arr = enc.transform(df_test[cat_col].astype(str)).toarray()
    df_test_ohe = pd.DataFrame(test_arr,columns=enc.get_feature_names())
    df_test = df_test.drop(columns=cat_col)
    df_test =pd.concat([df_test,df_test_ohe],axis=1)
    
    
    return df_train,df_test
        

def le_cat_col(df_train,df_test,cat_col):
    '''
    Label Encode all the categorical columns

    Parameters
    ----------
    df_train : Training DataFrame
    df_test : Test DataFrame
    cat_col : Categorical Columns

    Returns
    -------
    df_train : Training DataFrame
    df_test : Test DataFrame

    ''' 
    print('Label Encoding')
    for col in cat_col :
        print(col)
        le = LabelEncoder()
        le.fit(df_train[col].astype(str))
        df_test[col] = df_test[col].map(lambda s: 'other' if s not in le.classes_ else s)
        le_classes = le.classes_.tolist()
        bisect.insort_left(le_classes, 'other')
        le.classes_ = np.array(le_classes)
        
        df_train[col]  = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        
        
        
        
    
    return df_train,df_test

cat_col = ['Accomodation_Type','Reco_Insurance_Type','Is_Spouse','Health Indicator','Holding_Policy_Duration','Holding_Policy_Type',
           'Reco_Policy_Cat']


if __name__ == '__main__' :
    
    
    
    df_train = pd.read_csv(config.TRAIN_PATH_FEAT)
    df_test = pd.read_csv(config.TEST_PATH_FEAT)
    
    # One Hot Encoding
    df_train_ohe,df_test_ohe = ohe_cat_col(df_train,df_test,cat_col)

    df_train_ohe.to_csv(os.path.join(config.DATA_PATH,'train_ohe.csv'),index=False)
    df_test_ohe.to_csv(os.path.join(config.DATA_PATH,'test_ohe.csv'),index=False)     

    # Label Encoding
  #  df_train_le,df_test_le = le_cat_col(df_train,df_test,cat_col)

   # df_train_le.to_csv(os.path.join(config.DATA_PATH,'train_le.csv'),index=False)
   # df_test_le.to_csv(os.path.join(config.DATA_PATH,'test_le.csv'),index=False)     













