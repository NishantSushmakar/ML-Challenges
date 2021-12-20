import pandas as pd
from sklearn import model_selection
import os
import config

if __name__ == '__main__':
    
    df=pd.read_csv(config.TRAIN)
    
    df.loc[:,'Kfold'] = -1
    
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    targets = df['credit_card_default'].values
    
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    for f,(t_,v_) in enumerate(kf.split(X=df,y=targets)):
        
        df.loc[v_,'Kfold'] = f
        
        
    df.to_csv(os.path.join(config.LOC,'train_strat_woe.csv'),index=False)