import sys
import datetime
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
from skopt.space.space import Real, Integer
from models import HMMOOSSClassifier


def cv_1fold_iter(X):
    while True:
        yield np.arange(len(X)), []
        break
 

def main():
    outcome = sys.argv[1]#'Dementia'
    epoch_time = int(sys.argv[2])#30
    max_iter1 = 10
    max_iter2 = 100
    n_state_range = [5,15]
    batch_size = 8
    lr_reduce_patience = 3
    early_stop_patience = 10
    Ncv = 10
    class_weight = None # since using matched dataset
    random_state = 2023
    verbose = True###
    warmstart_log_folder = f'lightning_logs_warmstart_model_epochtime{epoch_time}s'
    log_folder = f'lightning_logs_epochtime{epoch_time}s'
    result_folder = f'results_new_{outcome}_epoch{epoch_time}s'
    os.makedirs(result_folder, exist_ok=True)
    
    df_feat = pd.read_csv(f'features_epoch{epoch_time}s.csv.zip')
    unique_sids = df_feat.HashID.unique()
    sids = df_feat.HashID.values
    Xnames = list(df_feat.columns)
    Xnames.remove('HashID')
    Xnames.remove('DOVshifted')
    Xnames.remove('SleepStage')
    Xnames.remove('EpochStartIdx')
    X = df_feat[Xnames].values
    df_y = pd.read_csv(f'../data/mastersheet_matched_{outcome}.csv')
    Y = df_y[f'Y_{outcome}'].values.astype(int)
    N = len(Y)
    print(f'N = {N}')
    print(f'Xnames = {Xnames}')
    
    ## get CV split
    
    cv_path = f'../step2_dementia_HMM_binary/cv_split_{outcome}_N={N}_epochtime{epoch_time}s_seed{random_state}.csv'
    df_cv = pd.read_csv(cv_path)
    print(f'CV assignment is read from {cv_path}')
       
    D = X.shape[1]
    binary_col_mask = np.array([set(X[:,i])==set([0,1]) for i in range(D)])
    
    warm_start_model_folder = f'warm_start_models_ok_epochtime{epoch_time}s'
    for ns in range(n_state_range[0], n_state_range[1]+1):
        save_path = os.path.join(warm_start_model_folder, f'n_state={ns}.pth')
        if os.path.exists(save_path):
            continue
        os.makedirs(warm_start_model_folder, exist_ok=True)
        print(f'warm starting n_state={ns}')
        model = HMMOOSSClassifier(
                    thres_bounds, D, Xnames=Xnames, n_components=ns,
                    lr=0.01, max_iter=max_iter1,
                    batch_size=batch_size, lr_reduce_patience=lr_reduce_patience,
                    early_stop_patience=early_stop_patience,
                    verbose=True, random_state=random_state+ns,
                    log_dir=warmstart_log_folder)
        X2 = [X[sids==sid] for sid in unique_sids]
        model.fit(X2, Y, separate=True)
        model.save(save_path, separate=True)
    
    yptes = np.zeros(N)
    models_cv = []
    for cvi in range(1,Ncv+1):
        print(f'\n\nnew model CV = {cvi}')
        trids = np.where(df_cv.CV!=cvi)[0]
        teids = np.where(df_cv.CV==cvi)[0]
        Xtr = [X[sids==sid] for sid in unique_sids[trids]]
        ytr = Y[trids]
        
        """
        model = HMMOOSSClassifier(
                thres_bounds, D, Xnames=Xnames,
                lr=0.01, max_iter=max_iter2, batch_size=batch_size,
                lr_reduce_patience=lr_reduce_patience, early_stop_patience=early_stop_patience,
                warm_start_model_folder=warm_start_model_folder,
                verbose=verbose, random_state=random_state,
                log_dir=log_folder)
        model = BayesSearchCV(model,
               {'n_components':Integer(n_state_range[0], n_state_range[1]),
                'C_l1': Real(1e-2, 1e0, 'log-uniform'),
                'C_Y':  Real(1e0, 1e+2, 'log-uniform'),
                'C_emission':  Real(1e-2, 1e1, 'log-uniform'),},
            scoring=None, cv=cv_1fold_iter(Xtr), n_points=10, n_iter=50,###
            n_jobs=1, verbose=10, random_state=random_state,
            )
        def on_step(opt_res):
            scores = -opt_res.func_vals
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            best_params = opt_res.x_iters[best_idx]
            print(f"current {len(scores)} scores: {scores}")
            print(f"current best score: {best_score}")
            print(f"current best params: {best_params}")
        model.fit(Xtr, ytr, callback=on_step)
        best_score = model.best_score_
        best_params = model.best_params_
        model = model.best_estimator_
        """
        models_ = []
        scores_ = []
        n_components = np.arange(n_state_range[0], n_state_range[1]+1)
        for ns in n_components:
            model_ = HMMOOSSClassifier(
                    thres_bounds, D, Xnames=Xnames,
                    lr=0.01, max_iter=max_iter2, batch_size=batch_size,
                    lr_reduce_patience=lr_reduce_patience, early_stop_patience=early_stop_patience,
                    random_state=random_state, verbose=verbose,
                    warm_start_model_folder=warm_start_model_folder,
                    n_components=ns, C_l1=0.1, C_Y=50, C_emission=0.1,
                    log_dir=log_folder)
            model_.fit(Xtr, ytr)
            models_.append(model_)
            scores_.append(model_.score(None,None))
            print(scores_)
        best_id = np.argmax(scores_)
        best_score = scores_[best_id]
        best_params = {'n_components':n_components[best_id]}
        model = models_[best_id]
            
        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'\n=={dt}==============================')
        print(f'[CV={cvi}]: NewModel best score = {best_score}')
        print(f'[CV={cvi}]: NewModel best params = {best_params}')
        print(f'[CV={cvi}]: NewModel X_thres = {model.X_thres_}')
        print(f'[CV={cvi}]: NewModel startprob = {model.startprob_}')
        print(f'[CV={cvi}]: NewModel transmat = {model.transmat_}')
        print(f'[CV={cvi}]: NewModel coef = {model.coef_}')
        print(f'[CV={cvi}]: NewModel path = {model.mytrainer.checkpoint_callbacks[0].best_model_path}')
        print('========================================\n')
        models_cv.append(model)
        
        Xte = [X[sids==sid] for sid in unique_sids[teids]]
        yptes[teids] = model.predict_proba(Xte)[:,1]
        model.save(os.path.join(result_folder, f'model_cv{cvi}.ckpt'))
        
    auc = roc_auc_score(Y, yptes)
    print(f'OVERALL CV AUC new = {auc}')
    
    # fit final model
    nc = int(round(np.median([m.n_components for m in models_cv])))
    C_l1 = np.median([m.C_l1 for m in models_cv])
    C_Y = np.median([m.C_Y for m in models_cv])
    C_emission = np.median([m.C_emission for m in models_cv])
    model_final = HMMOOSSClassifier(
            thres_bounds, D, Xnames=Xnames,
            n_components=nc, C_l1=C_l1, C_Y=C_Y, C_emission=C_emission,
            lr=0.01, max_iter=max_iter2, batch_size=batch_size,
            lr_reduce_patience=lr_reduce_patience, early_stop_patience=early_stop_patience,
            verbose=True, random_state=random_state,
            warm_start_model_folder=warm_start_model_folder,
            log_dir=log_folder)
    X2 = [X[sids==sid] for sid in unique_sids]
    model_final.fit(X2, Y)
    Zp = model_final.predict_proba_Z(X2)
    yp_final_train = model_final.predict_proba(X2)[:,1]
    
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, f'results.pickle'), 'wb') as ff:
        pickle.dump({
        'sids_te':sids, 'yte':Y, 'ypte':yptes, 'yp_final_train':yp_final_train,
        'Zp':Zp,
        }, ff)
    model_final.save(os.path.join(result_folder, 'model_final.ckpt'))
    
        
if __name__=='__main__':
    main()
    
