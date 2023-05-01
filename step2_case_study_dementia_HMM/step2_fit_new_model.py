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
    outcome = 'Dementia'
    epoch_time = 30
    max_iter1 = 10
    max_iter2 = 100###
    n_state_range = [3,15]###
    batch_size = 8
    lr_reduce_patience = 3
    early_stop_patience = 10
    Ncv = 10###
    class_weight = None # since using matched dataset
    random_state = 2023
    train_warm_start = False
    
    with open(f'dataset_{outcome}_epoch{epoch_time}s.pickle', 'rb') as ff:
        res = pickle.load(ff)
    sids = res['sids']
    X = res['X']
    Y = res['Y']
    Xnames = res['Xnames']
    N = len(Y)
    print(f'N = {N}')
    print(f'Xnames = {Xnames}')
    
    ## get CV split
    
    cv_path = f'cv_split_{outcome}_N={N}_seed{random_state}.csv'
    if os.path.exists(cv_path):
        df_cv = pd.read_csv(cv_path)
    else:
        cvf = StratifiedKFold(n_splits=Ncv, shuffle=True, random_state=random_state)
        cv_ids = np.zeros(N)
        for cvi, (trids, teids) in enumerate(cvf.split(np.zeros((len(X),1)), Y)):
            cv_ids[teids] = cvi+1
        df_cv = pd.concat([sids, pd.DataFrame(data={'Y':Y, 'CV':cv_ids})], axis=1)
        df_cv.to_csv(cv_path, index=False)
       
    ## train new model
    
    X2 = np.concatenate(X, axis=0)
    D = X2.shape[1]
    binary_col_mask = np.array([set(X2[:,i])==set([0,1]) for i in range(D)])
    X3 = X2[:,~binary_col_mask]
    X3[X3==0] = np.nan
    thres_bounds = np.zeros((2,D))
    thres_bounds[:,~binary_col_mask] = np.nanpercentile(X3, (5,95), axis=0)
    thres_bounds[0,binary_col_mask] = 0.49
    thres_bounds[1,binary_col_mask] = 0.51
    
    warm_start_model_folder = 'warm_start_models_ok'
    if train_warm_start:
        os.makedirs(warm_start_model_folder, exist_ok=True)
        for ns in range(n_state_range[0], n_state_range[1]+1):
            print(f'warm starting n_state={ns}')
            model = HMMOOSSClassifier(thres_bounds, D, Xnames=Xnames, n_components=ns, random_state=random_state+ns, lr=0.01, max_iter=max_iter1, batch_size=batch_size, lr_reduce_patience=lr_reduce_patience, early_stop_patience=early_stop_patience, verbose=True)
            model.fit(X, Y, separate=True)
            model.save(os.path.join(warm_start_model_folder, f'n_state={ns}.pth'), separate=True)
    
    yptes_new = np.zeros(N)
    models_new_cv = []
    for cvi in range(1,Ncv+1):
        print(f'\n\nnew model CV = {cvi}')
        trids = np.where(df_cv.CV!=cvi)[0]
        teids = np.where(df_cv.CV==cvi)[0]
        Xtr = [X[i] for i in trids]
        ytr = Y[trids]
        
        model_new = HMMOOSSClassifier(thres_bounds, D, Xnames=Xnames, random_state=random_state, lr=0.01, max_iter=max_iter2, batch_size=batch_size, lr_reduce_patience=lr_reduce_patience, early_stop_patience=early_stop_patience, verbose=False, warm_start_model_folder=warm_start_model_folder)
        model_new = BayesSearchCV(model_new,
               {'n_components':Integer(n_state_range[0], n_state_range[1]),
                'C_l1': Real(1e-2, 1e1, 'log-uniform'),
                'C_Y':  Real(1e0, 1e+2, 'log-uniform'),
                'C_emission':  Real(1e-3, 1e0, 'log-uniform'),},
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
        model_new.fit(Xtr, ytr, callback=on_step)
            
        dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('\n========================================')
        print(f'[{dt}] CV = {cvi}: NewModel best score = {model_new.best_score_}')
        print(f'[{dt}] CV = {cvi}: NewModel best params = {model_new.best_params_}')
        print(f'[{dt}] CV = {cvi}: NewModel X_thres = {model_new.best_estimator_.X_thres_}')
        print(f'[{dt}] CV = {cvi}: NewModel startprob = {model_new.best_estimator_.startprob_}')
        print(f'[{dt}] CV = {cvi}: NewModel transmat = {model_new.best_estimator_.transmat_}')
        print(f'[{dt}] CV = {cvi}: NewModel coef = {model_new.best_estimator_.coef_}')
        print(f'[{dt}] CV = {cvi}: NewModel path = {model_new.best_estimator_.mytrainer.checkpoint_callbacks[0].best_model_path}')
        print('========================================\n')
        models_new_cv.append(model_new.best_estimator_)
        
        Xte = [X[i] for i in teids]
        yptes_new[teids] = model_new.predict_proba(Xte)[:,1]
        
    auc = roc_auc_score(Y, yptes_new)
    print(f'OVERALL CV AUC new = {auc}')
    
    # fit final model
    import pdb;pdb.set_trace()
    nc = int(round(np.median([m.n_components for m in models_new_cv])))
    C_l1 = np.median([m.C_l1 for m in models_new_cv])
    C_Y = np.median([m.C_Y for m in models_new_cv])
    C_emission = np.median([m.C_emission for m in models_new_cv])
    model_new_final = HMMOOSSClassifier(thres_bounds, D, Xnames=Xnames, n_components=nc, C_l1=C_l1, C_Y=C_Y, C_emission=C_emission, random_state=random_state, lr=0.01, max_iter=max_iter2, batch_size=batch_size, lr_reduce_patience=lr_reduce_patience, early_stop_patience=early_stop_patience, verbose=True, warm_start_model_folder=warm_start_model_folder)
    model_new_final.fit(X,Y)
    Zp = model_new_final.predict_proba_Z(X)
    yp_new_final_train = model_new_final.predict_proba(X)[:,1]
    
    result_folder = f'results_{outcome}_epoch{epoch_time}s'
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, f'results_new.pickle'), 'wb') as ff:
        pickle.dump({
        'sids_te':sids, 'yte':Y, 'ypte_new':yptes_new, 'yp_new_final_train':yp_new_final_train,
        'Zp':Zp,
        }, ff)
    for mi, m in enumerate(models_new_cv):
        m.save(os.path.join(result_folder, f'models_new_cv{mi+1}'))
    model_new_final.save(os.path.join(result_folder, 'model_new_final'))
    
        
if __name__=='__main__':
    main()
    
