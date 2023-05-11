import os
import shutil
import pickle
import zipfile
import numpy as np
from scipy.stats import mode
#from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV, gp_minimize
from skopt.space.space import Real, Integer
from myhmmclassifier import MyHMMClassifier



def reset_warm_start(folder):
    os.makedirs(folder, exist_ok=True)
    for nc in range(1,100):
        warm_start_path = os.path.join(folder, f'n_components={nc}.ckpt')
        if os.path.exists(warm_start_path):
            os.remove(warm_start_path)
            

def main():
    outcome = 'Dementia'
    epoch_time = 30
    with  open(f'dataset_{outcome}_epoch{epoch_time}s.pickle', 'rb') as ff:
        res = pickle.load(ff)
    sids = res['sids']
    print(f'N = {len(sids)}')
    X = res['X']
    S = res['S']
    Y = res['Y']
    Xnames = res['Xnames']
    random_state = 2023
    Ncv = 10
    n_jobs = 14
    class_weight = None # since using matched dataset
    
    X2 = np.concatenate(X, axis=0)
    binary_col_mask = np.array([set(X2[:,i])==set([0,1]) for i in range(X2.shape[1])])
    X3 = X2[:,~binary_col_mask]
    X3[X3==0] = np.nan
    thres_bounds = np.nanpercentile(X3, (5,95), axis=0)
    warm_start_model_folder = 'warm_start_models'
    
    #teids_cv = []
    sids_te = []
    ytes = []
    yptes_new = []
    yptes_aasm = []
    models_new_cv = []
    models_aasm_cv = []
    cvf = StratifiedKFold(n_splits=Ncv, shuffle=True, random_state=random_state)
    for cvi, (trids, teids) in enumerate(cvf.split(np.zeros((len(X),1)), Y)):
        #teids_cv.append(teids)
        sids_te.append(sids.iloc[teids])
        print(f'CV = {cvi}')
        Xtr = [X[i] for i in trids]
        Str = [S[i] for i in trids]
        ytr = Y[trids]
        
        ## train new model
        
        reset_warm_start(warm_start_model_folder)
        def mycallback(res):
            if np.all(res.func_vals[-1]<res.func_vals[:-1]):
                if os.path.exists('current_best.zip'):
                    os.remove('current_best.zip')
                os.rename('tmp.zip', 'current_best.zip')
            else:
                os.remove('tmp.zip')
        def loss(params):
            n_components = params[0]
            thres = params[1:]
            model_new = MyHMMClassifier(thres=thres, n_components=n_components, binary_col_mask=binary_col_mask, random_state=random_state, verbose=False, n_jobs=n_jobs, warm_start_model_folder=warm_start_model_folder)
            model_new.fit(Xtr, ytr)
            model_new.save('tmp.zip')
            return -model_new.model_Y_best_cv_score_
        opt_res = gp_minimize(loss,
            [Integer(3,15)]+[Real(thres_bounds[0,ii],thres_bounds[1,ii]) for ii in range(thres_bounds.shape[1])],
            n_calls=3, n_initial_points=2, initial_point_generator='lhs', callback=mycallback,
            random_state=random_state, verbose=True, n_jobs=1)
        model_new = MyHMMClassifier.load('current_best.zip')
        if os.path.exists('current_best.zip'): os.remove('current_best.zip')
        models_new_cv.append(model_new)
        print(f'best_scores(HMM) = {model_new.model_Y_best_cv_score_}')
        print(f'new model coef = {model_new.model_Y.coef_}')
        
        Xte = [X[i] for i in teids]
        import pdb;pdb.set_trace()
        ypte_new = model_new.predict_proba(Xte)[:,1]
        
        ## train AASM model
        
        model_aasm = LogisticRegression(penalty='elasticnet', tol=0.0001, class_weight=class_weight, random_state=random_state, solver='saga', max_iter=10000)#, verbose=True)
        model_aasm = BayesSearchCV(model_aasm,
               {'C': (1e-3, 1e+3, 'log-uniform'),
                'l1_ratio': (0.01, 0.99, 'uniform'),},
            scoring='roc_auc', cv=Ncv, n_points=1, n_iter=2,
            n_jobs=n_jobs, verbose=False, random_state=random_state,
            )
        Zhist = np.array([[np.mean(z==zz) for zz in range(1,5-1+1)] for z in Str])
        model_aasm.fit(Zhist, ytr)
        print(f'best_scores(AASM) = {model_aasm.best_score_}')
        print(f'AASM model coef = {model_aasm.best_estimator_.coef_}')
        models_aasm_cv.append(model_aasm.best_estimator_)
        
        Ste = [S[i] for i in teids]
        Zhist = np.array([[np.mean(z==zz) for zz in range(1,5-1+1)] for z in Ste])
        ypte_aasm = model_aasm.predict_proba(Zhist)[:,1]
        
        ytes.append(Y[teids])
        yptes_new.append(ypte_new)
        yptes_aasm.append(ypte_aasm)

    yte = np.concatenate(ytes)
    ypte_new = np.concatenate(yptes_new)
    ypte_aasm = np.concatenate(yptes_aasm)
    
    #TODO fit final model
    nc = mode([m.n_components for m in models_new_cv]).mode[0]
    C = np.median([m.n_components for m in models_new_cv])
    l1_ratio = np.median([m.l1_ratio for m in models_new_cv])
    model_new_final = MyHMMClassifier(thres_bounds, binary_col_mask, n_components=nc, C=C, l1_ratio=l1_ratio, class_weight=class_weight, random_state=random_state, verbose=False)
    model_new_final.fit(X)
    Zp = model_new_final.predict_proba_Z(X, return_list=True)
    #yp_new = model_new_final.predict_proba(X)[:,1]
         
    C = np.median([m.n_components for m in models_aasm_cv])
    l1_ratio = np.median([m.l1_ratio for m in models_aasm_cv])
    model_Y_aasm_final = LogisticRegression(
        penalty='elasticnet', tol=0.0001, class_weight=class_weight,
        random_state=random_state, solver='saga', max_iter=10000,
        C=C, l1_ratio=l1_ratio)
    Zhist = np.array([[np.mean(z==zz) for zz in range(1,5-1+1)] for z in S])
    model_aasm_final.fit(Zhist, Y)
    #yp_aasm = model_aasm_final.predict_proba(X)[:,1]
    
    result_folder = f'results_{outcome}'
    os.makedirs(result_folder, exist_ok=True)
    with open(os.path.join(result_folder, f'results_{outcome}.pickle'), 'wb') as ff:
        pickle.dump({
        'sids_te':sids_te, 'yte':ytes, 'ypte_new':yptes_new, 'ypte_aasm':yptes_aasm,
        'Zp':Zp,
        'models_aasm_cv':models_aasm_cv, 'model_aasm_final':model_aasm_final
        }, ff)
    for mi, m in enumerate(models_new_cv):
        m.save(os.path.join(result_folder, f'models_new_cv{mi+1}'))
    model_new_final.save(os.path.join(result_folder, 'model_new_final'))
    #shutil.make_archive(result_folder, 'zip', result_folder)
    #shutil.rmtree(result_folder)
    
        
if __name__=='__main__':
    main()
    
