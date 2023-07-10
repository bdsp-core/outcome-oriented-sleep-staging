import datetime
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV


def get_features(S):
    stages = [1,2,3,4,5]
    tst = np.sum(np.in1d(S, stages))*30/3600

    stage_times_hour = []
    stage_perc = []
    for s in stages:
        ids = S==s
        stage_times_hour.append(np.sum(ids)*30/3600)
        stage_perc.append(np.mean(ids))

    cm = np.zeros((len(stages), len(stages)))+1
    for s1, s2 in zip(S[:-1], S[1:]):
        if s1 in stages and s2 in stages:
            s1_ = int(s1-1)
            s2_ = int(s2-1)
            cm[s1_,s2_] += 1
    cm = cm/cm.sum(axis=1, keepdims=True)
    cm = cm.flatten()

    sleep_ids = np.where(np.in1d(S, [1,2,3,4]))[0]
    if len(sleep_ids)>0:
        sl = sleep_ids[0]*30/60
    else:
        sl = len(S)*30/60

    rem_ids = np.where(np.in1d(S, [4]))[0]
    if len(rem_ids)>0:
        rl = rem_ids[0]*30/60
    else:
        rl = len(S)*30/60

    if len(sleep_ids)>0:
        waso = (S[sleep_ids[0]:sleep_ids[-1]+1]==5).sum()*30/60
    else:
        waso = len(S)*30/60

    #return np.r_[tst, stage_times_hour, stage_perc, cm, sl, rl, waso]
    return stage_perc
            

def main():
    outcome = 'Dementia'
    epoch_time = 30
    Ncv = 10
    class_weight = None # since using matched dataset
    random_state = 2023
    result_folder = f'results_aasm_{outcome}_epoch{epoch_time}s'
    
    df_feat = pd.read_csv(f'features_epoch{epoch_time}s.csv.zip')
    sids = df_feat.HashID.values
    unique_sids = pd.unique(sids)
    S = df_feat.SleepStage.values
    S = [S[sids==x] for x in unique_sids]
    df_y = pd.read_csv(f'../data/mastersheet_matched_{outcome}.csv')
    Y = df_y[f'Y_{outcome}'].values.astype(int)
    N = len(Y)
    print(f'N = {N}')
    
    ## get CV split
    
    cv_path = f'cv_split_{outcome}_N={N}_epochtime{epoch_time}s_seed{random_state}.csv'
    df_cv = pd.read_csv(cv_path)
        
    ## train AASM model
        
    yptes_aasm = np.zeros(N)
    models_aasm_cv = []
    for cvi in range(1,Ncv+1):
        print(f'AASM model CV = {cvi}')
        trids = np.where(df_cv.CV!=cvi)[0]
        teids = np.where(df_cv.CV==cvi)[0]
        Str = [S[i] for i in trids]
        ytr = Y[trids]
        
        model_aasm = LogisticRegression(penalty='l1', tol=0.001, class_weight=class_weight, random_state=random_state, solver='liblinear', max_iter=10000)
        model_aasm = BayesSearchCV(model_aasm,
               {'C': (1e0, 1e+3, 'log-uniform'),
                #'l1_ratio': (0.01, 0.99, 'uniform'),
                },
            scoring='roc_auc', cv=Ncv, n_points=10, n_iter=50,###
            n_jobs=8, verbose=False, random_state=random_state,
            )

        Xtr = np.array([get_features(s) for s in Str])
        model_aasm.fit(Xtr, ytr)
        
        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{dt}] CV = {cvi}: AASM best score = {model_aasm.best_score_}')
        print(f'[{dt}] CV = {cvi}: AASM best param = {model_aasm.best_params_}')
        print(f'[{dt}] CV = {cvi}: AASM model coef = {model_aasm.best_estimator_.coef_}')
        models_aasm_cv.append(model_aasm.best_estimator_)
        
        Ste = [S[i] for i in teids]
        Xte = np.array([get_features(s) for s in Ste])
        yptes_aasm[teids] = model_aasm.predict_proba(Xte)[:,1]

    import pdb;pdb.set_trace()
    auc = roc_auc_score(Y, yptes_aasm)
    print(f'OVERALL CV AUC AASM = {auc}')
    
    # fit final model
    best_C = np.median([m.C for m in models_aasm_cv])
    model_aasm_final = LogisticRegression(penalty='l1', C=best_C, tol=0.001, class_weight=class_weight, random_state=random_state, solver='liblinear', max_iter=10000)
    X = np.array([get_features(s) for s in S])
    model_aasm_final.fit(X, Y)
    print(f'Final AASM coef = {model_aasm_final.coef_}')
    
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, f'results_aasm.pickle'), 'wb') as ff:
        pickle.dump({
        'sids_te':sids, 'yte':Y, 'ypte_aasm':yptes_aasm,
        'models_aasm_cv':models_aasm_cv, 'model_aasm_final':model_aasm_final
        }, ff)
    
        
if __name__=='__main__':
    main()
    
