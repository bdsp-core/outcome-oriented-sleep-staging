import sys
import datetime
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
from skopt.space.space import Real, Integer
from models import HMMOOSSClassifier


def cv_1fold_iter(X):
    while True:
        yield np.arange(len(X)), []
        break
 

def main():
    ## set parameters

    epoch_time = int(sys.argv[1])#30
    n_state_range = [10,25]
    max_iter1 = 10; max_iter2 = 100
    batch_size = 8
    lr_reduce_patience = 3
    early_stop_patience = 10
    Ncv = 10
    class_weight = None # since using matched dataset
    random_state = 2023
    verbose = True###
    warmstart_log_folder = f'warmstart_log_epochtime{epoch_time}s'
    warmstart_model_folder = f'warmstart_model_epochtime{epoch_time}s'
    log_folder = f'log_epochtime{epoch_time}s'
    result_folder = f'results_new_multiple_outcomes_epoch{epoch_time}s'
    os.makedirs(result_folder, exist_ok=True)

    ## get variables

    # get X
    df_feat = pd.read_csv(f'../data/features_epoch{epoch_time}s.csv.zip')
    unique_sids = df_feat.HashID.unique()
    sids = df_feat.HashID.values
    Xnames = list(df_feat.columns)
    Xnames.remove('HashID')
    Xnames.remove('DOVshifted')
    Xnames.remove('SleepStage')
    Xnames.remove('EpochStartIdx')
    X = df_feat[Xnames].values
    bin_mask = np.array([set(X[:,i])==set([0,1]) for i in range(X.shape[1])])
    X = [X[sids==sid] for sid in unique_sids]
    print(f'N = {len(X)}')
    print(f'Xnames = {Xnames}')

    # get S
    S = df_feat.SleepStage.values
    S2 = np.zeros((len(S),3))
    S2[np.in1d(S,[1,2,3]), 0] = 1  # NREM
    S2[S==4, 1] = 1  # REM
    S2[S==5, 2] = 1  # W
    S = [S2[sids==sid] for sid in unique_sids]
    XS = [(x,s) for x,s in zip(X,S)]

    # get Y
    df_y = pd.read_csv(f'../data/mastersheet_matched_Dementia_alloutcomes.csv')
    ycols = ['Y_Dementia',
        'Y_Hypertension',
        'Y_Depression',
        'Y_Atrial_Fibrillation',
        'Y_Myocardial_Infarction',]
    Y = df_y[ycols].values#.astype(int)
    
    ## get CV split
    
    cv_path = f'../data/cv_split_Dementia_N={len(X)}_epochtime{epoch_time}s_seed{random_state}.csv'
    df_cv = pd.read_csv(cv_path)

    ## warm start
    
    for ns in range(n_state_range[0], n_state_range[1]+1):
        save_path = os.path.join(warmstart_model_folder, f'n_state={ns}.pth')
        if os.path.exists(save_path):
            continue
        os.makedirs(warmstart_model_folder, exist_ok=True)
        print(f'warm starting n_state={ns}')
        model = HMMOOSSClassifier(
            Xnames, bin_mask, n_components=ns, class_weight=class_weight,
            lr=0.01, max_iter=max_iter1, batch_size=batch_size,
            lr_reduce_patience=lr_reduce_patience, early_stop_patience=early_stop_patience,
            verbose=True, random_state=random_state+ns,
            log_dir=warmstart_log_folder)
        model.fit(XS, Y, separate=True)
        model.save(save_path, separate=True)

    ## fit CV models
    
    yptes = np.zeros((N, len(ycols)))
    models_cv = []
    for cvi in range(1,Ncv+1):
        print(f'\n\nnew model CV = {cvi}')
        trids = np.where(df_cv.CV!=cvi)[0]
        teids = np.where(df_cv.CV==cvi)[0]
        Xtr = [X[sids==sid] for sid in unique_sids[trids]]
        ytr = Y[trids]
        
        """
        model = HMMOOSSClassifier(
                Xnames, bin_mask, class_weight=class_weight,
                lr=0.01, max_iter=max_iter2, batch_size=batch_size,
                lr_reduce_patience=lr_reduce_patience, early_stop_patience=early_stop_patience,
                warmstart_model_folder=warmstart_model_folder,
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
        scores_each_outcome_ = []
        n_components = np.arange(n_state_range[0], n_state_range[1]+1)
        for ns in n_components:
            model_ = HMMOOSSClassifier(
                    Xnames, bin_mask, class_weight=class_weight,
                    lr=0.01, max_iter=max_iter2, batch_size=batch_size,
                    lr_reduce_patience=lr_reduce_patience, early_stop_patience=early_stop_patience,
                    random_state=random_state, verbose=verbose,
                    warmstart_model_folder=warmstart_model_folder,
                    n_components=ns, C_l1=0.1, C_Y=50, C_emission=0.1,
                    log_dir=log_folder)
            import pdb;pdb.set_trace()
            model_.fit(Xtr, ytr)
            models_.append(model_)
            a, b = model_.score(return_components=True)
            scores_.append(a)
            scores_each_outcome_.append(b)
            print(scores_)
            print(scores_each_outcome_)
        best_id = np.argmax(scores_)
        best_score = scores_[best_id]
        best_params = {'n_components':n_components[best_id]}
        model = models_[best_id]
            
        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'\n=={dt}==============================')
        print(f'[CV={cvi}]: NewModel best score = {best_score}')
        print(f'[CV={cvi}]: NewModel best score = {scores_each_outcome_[best_id]}')
        print(f'[CV={cvi}]: NewModel best params = {best_params}')
        print(f'[CV={cvi}]: NewModel startprob = {model.startprob_}')
        print(f'[CV={cvi}]: NewModel transmat = {model.transmat_}')
        print(f'[CV={cvi}]: NewModel coef = {model.coef_}')
        print(f'[CV={cvi}]: NewModel path = {model.mytrainer.checkpoint_callbacks[0].best_model_path}')
        print('========================================\n')
        models_cv.append(model)
        
        Xte = [X[sids==sid] for sid in unique_sids[teids]]
        yptes[teids] = model.predict_proba(Xte)#[:,1]
        model.save(os.path.join(result_folder, f'model_cv{cvi}.ckpt'))
        
    for yi, ycol in enumerate(ycols):
        ids = ~np.isnan(Y[:,yi])
        auc = roc_auc_score(Y[ids][:,yi], yptes[ids][:,yi])
        print(f'{ycol}: OVERALL CV AUC new = {auc}')
    
    ## fit final model

    nc = int(round(np.median([m.n_components for m in models_cv])))
    C_l1 = np.median([m.C_l1 for m in models_cv])
    C_Y = np.median([m.C_Y for m in models_cv])
    C_emission = np.median([m.C_emission for m in models_cv])
    model_final = HMMOOSSClassifier(
            Xnames, bin_mask, class_weight=class_weight,
            n_components=nc, C_l1=C_l1, C_Y=C_Y, C_emission=C_emission,
            lr=0.01, max_iter=max_iter2, batch_size=batch_size,
            lr_reduce_patience=lr_reduce_patience, early_stop_patience=early_stop_patience,
            verbose=True, random_state=random_state,
            warmstart_model_folder=warmstart_model_folder,
            log_dir=log_folder)
    model_final.fit(X, Y)
    #model_final.postprocess_states(X, [sleep_stages[sids==sid] for sid in unique_sids], good_state_thres=1800/epoch_time)
    Zp = model_final.predict_proba_Z(X)
    yp_final_train = model_final.predict_proba(X)#[:,1]

    ## save results

    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, f'results.pickle'), 'wb') as ff:
        pickle.dump({
            'sids_te':sids, 'outcomes':ycols,
            'yte':Y, 'ypte':yptes, 'yp_final_train':yp_final_train,
            'Zp':Zp, }, ff)
    model_final.save(os.path.join(result_folder, 'model_final.ckpt'))
    
        
if __name__=='__main__':
    main()
    
