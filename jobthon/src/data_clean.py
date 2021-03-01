# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:46:55 2021

@author: nishant
"""
import pandas as pd
import numpy as np
import os 
import config

def data_cleaning(df):
    ''' 
    Function Handles the missing value
    
    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    Cleaned DataFrame

    '''
    col_with_missing = ['Holding_Policy_Type','Health Indicator']
   
    for col in col_with_missing :
        df.loc[df[col].isnull()==True,col] = 'Unknown'
        
        
    df.loc[df['Holding_Policy_Duration'].isnull()==True,'Holding_Policy_Duration'] = '0'
   
    return df 




if __name__ == '__main__' : 
    
    df = pd.read_csv(config.TEST_PATH)
    df = data_cleaning(df)
    
    df.to_csv(os.path.join(config.DATA_PATH,'cleaned_test.csv'),index=False)