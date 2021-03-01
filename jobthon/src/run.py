# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:11:35 2021

@author: nishant
"""
import config
import os
import joblib
import pandas as pd
import model_dispatch
from sklearn.metrics import roc_auc_score,accuracy_score
import numpy as np
import optimize

def run(fold,model1,model2,model3):
    '''
    train and caluclate roc for the validation data for the given fold  

    Parameters
    ----------
    df : dataframe
    fold : Fold to test on

    Returns
    -------
    None.

    '''
    features_to_remove = ['ID','Response','kfold']
    df = pd.read_csv(config.FINAL_TRAIN)
        
    
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)
    
    
    x_train = df_train.drop(columns=features_to_remove,axis=1)
    y_train = df_train.Response.values
    
    x_valid = df_valid.drop(columns=features_to_remove,axis=1)
    y_valid = df_valid.Response.values
    
    
    
    clf1 = model_dispatch.models[model1]
    clf2 = model_dispatch.models[model2]
    clf3 = model_dispatch.models[model3]
    
    clf1.fit(x_train,y_train)
    clf2.fit(x_train,y_train)
    clf3.fit(x_train,y_train)
    
    pred1= clf1.predict_proba(x_train)[:,1]
    pred2= clf2.predict_proba(x_train)[:,1]
    pred3= clf3.predict_proba(x_train)[:,1]
    
    fold_preds = np.column_stack((
             pred1,pred2,pred3
        ))
    
    pred1_val= clf1.predict_proba(x_valid)[:,1]
    pred2_val= clf2.predict_proba(x_valid)[:,1]
    pred3_val= clf3.predict_proba(x_valid)[:,1]
    
    val_preds = np.column_stack((
        pred1_val,pred2_val,pred3_val
        ))
    print('Optimization Started')
    print(fold_preds.shape)
    opt = optimize.optimizeAUC()
    opt.fit(fold_preds,y_train)
    opt_pred_val = opt.predict(val_preds)
    opt_pred_train = opt.predict(fold_preds)
    
    print(f'AUC Score Train:{roc_auc_score(y_train,opt_pred_train)}')
    print(f'AUC Score Validation:{roc_auc_score(y_valid,opt_pred_val)}')
    print(f'coefficient  : {opt.coef_}')
    
    
    
   # roc_auc_train = roc_auc_score(y_train,clf.predict_proba(x_train)[:,1])
   # roc_auc_valid = roc_auc_score(y_valid,clf.predict_proba(x_valid)[:,1])
   # print(f'Train Accuracy : {accuracy_score(y_train,clf.predict(x_train))}')
   # print(f'Validation Accuracy : {accuracy_score(y_valid,clf.predict(x_valid))}')
   # print(f'Train ROC_AUC_SCORE:{roc_auc_train}')
   # print(f'Validation ROC_AUC_SCORE:{roc_auc_valid}')
    
  #  joblib.dump(clf,os.path.join(config.MODEL_PATH,f"{model}_{fold}.bin"))
    
    
if __name__ == '__main__' :
    
    for i in range(5):
            print(i)
            
            run(i,'xgb',"random_forest_gini",'random_forest_entropy')
                
    
    
    
    
    
    
    
    
    