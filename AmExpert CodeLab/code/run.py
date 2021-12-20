# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 19:30:54 2021

@author: nishant
"""
import config
import os
import joblib
import pandas as pd
import models_dispatch
from sklearn.metrics import f1_score
import numpy as np
from scipy import stats

def run(df,test_df,fold,model):
    '''
    train and caluclate f1 score for the validation data for the given fold  
    Parameters
    ----------
    df : dataframe
    fold : Fold to test on
    Returns
    -------
    None.
    '''
    
    features_to_remove = ['customer_id','name','Kfold','credit_card_default']
    
    df_train = df[df.Kfold!=fold].reset_index(drop=True)
    df_valid = df[df.Kfold==fold].reset_index(drop=True)
    
    
    x_train = df_train.drop(columns=features_to_remove,axis=1)
    y_train = df_train.credit_card_default.values
    
    x_valid = df_valid.drop(columns=features_to_remove,axis=1)
    y_valid = df_valid.credit_card_default.values
    
    
    clf = models_dispatch.models[model]
    if model=='cat':
        clf.fit(x_train,y_train,verbose=0,eval_set=[(x_valid,y_valid)])
    else:
        clf.fit(x_train,y_train)
    # verbose=0,eval_set=[(x_valid,y_valid)],early_stopping_rounds=100
    pred= clf.predict(x_train)
    pred_val= clf.predict(x_valid)
    
    print('F1 Score Train:',f1_score(y_train,pred,average='macro'))
    print('F1 Score Validation:',f1_score(y_valid,pred_val,average='macro'))
    
    #print('AUC Score Train:',roc_auc_score(y_train,clf.predict_proba(x_train)[:,1]))
    #print('AUC Score Validation:',roc_auc_score(y_valid,clf.predict_proba(x_valid)[:,1]))
    
    joblib.dump(clf,os.path.join(config.MODEL_DIR,f"{model}_{fold}.bin"))
    
    # Prediction on Test Data
    test = test_df.drop(columns=['customer_id','name'])
    
    return clf.predict(test)
    
    
if __name__ == '__main__' :
    
    train_df = pd.read_csv(config.TRAIN)
    test_df = pd.read_csv(config.TEST)    
    
    model = 'cat'
    scores = []
    print(model)
    for i in range(5):
        print('Fold ',i)
        scores.append(run(train_df,test_df,i,model))
                
    
    scores = stats.mode(scores,axis=0)
    scores = np.array(scores[0]).reshape(-1)
    submission = pd.DataFrame({'customer_id':test_df.customer_id,'credit_card_default':scores})
    submission.to_csv(os.path.join(config.SUBMIT,f'submission_{model}_avg.csv'),index=False)
    
    