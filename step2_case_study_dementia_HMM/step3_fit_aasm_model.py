import datetime
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
            

def main():
    outcome = 'Dementia'
    epoch_time = 30
    Ncv = 10
    class_weight = None # since using matched dataset
    random_state = 2023
    
    with open(f'dataset_{outcome}_epoch{epoch_time}s.pickle', 'rb') as ff:
        res = pickle.load(ff)
    sids = res['sids']
    X = res['X']
    S = res['S']
    Y = res['Y']
    Xnames = res['Xnames']
    N = len(Y)
    print(f'N = {N}')
    
    ## get CV split
    
    cv_path = f'cv_split_{outcome}_N={N}_seed{random_state}.csv'
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
            n_jobs=14, verbose=False, random_state=random_state,
            )
        Zhist = np.array([[np.mean(z==zz) for zz in range(1,5-1+1)] for z in Str])
        model_aasm.fit(Zhist, ytr)
        
        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{dt}] CV = {cvi}: AASM best score = {model_aasm.best_score_}')
        print(f'[{dt}] CV = {cvi}: AASM best param = {model_aasm.best_params_}')
        print(f'[{dt}] CV = {cvi}: AASM model coef = {model_aasm.best_estimator_.coef_}')
        models_aasm_cv.append(model_aasm.best_estimator_)
        
        Ste = [S[i] for i in teids]
        Zhist = np.array([[np.mean(z==zz) for zz in range(1,5-1+1)] for z in Ste])
        yptes_aasm[teids] = model_aasm.predict_proba(Zhist)[:,1]

    auc = roc_auc_score(Y, yptes_aasm)
    print(f'OVERALL CV AUC AASM = {auc}')
    
    # fit final model
    best_C = np.median([m.C for m in models_aasm_cv])
    model_aasm_final = LogisticRegression(penalty='l1', C=best_C, tol=0.001, class_weight=class_weight, random_state=random_state, solver='liblinear', max_iter=10000)
    Zhist = np.array([[np.mean(z==zz) for zz in range(1,5-1+1)] for z in S])
    model_aasm_final.fit(Zhist, Y)
    print(f'Final AASM coef = {model_aasm_final.coef_}')
    
    result_folder = f'results_{outcome}_epoch{epoch_time}s'
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, f'results_aasm.pickle'), 'wb') as ff:
        pickle.dump({
        'sids_te':sids, 'yte':Y, 'ypte_aasm':yptes_aasm,
        'models_aasm_cv':models_aasm_cv, 'model_aasm_final':model_aasm_final
        }, ff)
    
        
if __name__=='__main__':
    main()
    
