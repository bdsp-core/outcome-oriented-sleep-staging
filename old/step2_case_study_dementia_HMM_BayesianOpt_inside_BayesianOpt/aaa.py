import os
import shutil
import pickle
import zipfile
import numpy as np
from scipy.stats import mode
from scipy.optimize import minimize
#from hmmlearn.hmm import CategoricalHMM
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from skopt import BayesSearchCV, gp_minimize
from skopt.space.space import Real, Integer
from myhmmlearn import BernoulliHMM


class MyHMMClassifier(BaseEstimator, ClassifierMixin):
    """
    """
    def __init__(self, thres=None, binary_col_mask=None, n_components=2, random_state=None, verbose=False, n_jobs=1, hmm_n_iter=200, class_weight=None, warm_start_model_folder='warm_start_models'):
        self.thres = thres
        self.binary_col_mask = binary_col_mask
        self.n_components = n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.hmm_n_iter = hmm_n_iter
        self.class_weight = class_weight
        self.warm_start_model_folder = warm_start_model_folder
        
    def _preprocess_X(self, X):   
        T = [len(x) for x in X]
        to_list_ids = np.r_[0,np.cumsum(T)]
        Xb = np.concatenate(X)
        if self.binary_col_mask is None:
            self.binary_col_mask = np.zeros(Xb.shape[1],dtype=bool)
        if self.thres is None:
            self.thres = np.nanmean(Xb[:,~self.binary_col_mask], axis=0)
        Xb[:,~self.binary_col_mask] = (Xb[:,~self.binary_col_mask]>self.thres).astype(float)
        Xb = Xb.astype(int)
        return Xb, T, to_list_ids
        
    def fit(self, X, y):
        Xb, T, to_list_ids = self._preprocess_X(X)
        
        warm_start_path = os.path.join(self.warm_start_model_folder, f'n_components={self.n_components}.ckpt')
        if os.path.exists(warm_start_path):
            self.model_Z = BernoulliHMM.load_from_checkpoint(warm_start_path)
            self.model_Z.hparams.lr = 0.001
        else:
            print(f'first time running n_components={self.n_components}')
            self.model_Z = BernoulliHMM(n_components=self.n_components, n_features=Xb.shape[1], batch_size=8, lr=0.01, n_iter=self.hmm_n_iter, tol=0.1, verbose=self.verbose)
        self.model_Z.fit(Xb, T)
        if not os.path.exists(warm_start_path):
            self.model_Z.trainer.save_checkpoint(warm_start_path)
            print('finish')
        
        Z = self.model_Z.predict(Xb, T)
        Z = [Z[to_list_ids[i]:to_list_ids[i+1]] for i in range(len(to_list_ids)-1)]
        Zhist = np.array([[np.mean(z==zz) for zz in range(self.n_components-1)] for z in Z])

        # logistic regression
        Ncv = 5
        self.model_Y = LogisticRegression(penalty='elasticnet', tol=0.0001, class_weight=self.class_weight, random_state=self.random_state+100, solver='saga', max_iter=10000, verbose=False)
        self.model_Y = BayesSearchCV(self.model_Y, {
                'C': (1e-3, 1e+3, 'log-uniform'),
                'l1_ratio': (0.01, 0.99, 'uniform'),
            },
            scoring='roc_auc', cv=Ncv, n_points=10, n_iter=50,
            n_jobs=self.n_jobs, verbose=self.verbose, random_state=self.random_state,
            )
        self.model_Y.fit(Zhist, y)
        self.model_Y_best_cv_score_ = self.model_Y.best_score_
        self.model_Y = self.model_Y.best_estimator_
        
        # calibration
        if self.class_weight == 'balanced':
            self.model_Y = CalibratedClassifierCV(base_estimator=self.model_Y, method='sigmoid', cv='prefit')
            self.model_Y.fit(Zhist, y)
            yp = self.model_Y.predict_proba(Zhist)[:,1]
            fpr, tpr, tt = roc_curve(y, yp)
            self.best_yp_thres = tt[np.argmin(fpr**2+(1-tpr)**2)]
        else:
            self.best_yp_thres = 0.5
        return self
    
    def decision_function(self, X, return_Zhist=False):
        Xb, T, to_list_ids = self._preprocess_X(X)
    
        Z = self.model_Z.predict(Xb, T)
        Z = [Z[to_list_ids[i]:to_list_ids[i+1]] for i in range(len(to_list_ids)-1)]
        Zhist = np.array([[np.mean(z==zz) for zz in range(self.n_components-1)] for z in Z])

        if self.class_weight == 'balanced':
            d = self.model_Y.base_estimator.decision_function(Zhist)
        else:
            d = self.model_Y.decision_function(Zhist)
        if return_Zhist:
            return d, Zhist
        else:
            return d
        
    def predict_proba(self, X):
        _, Zhist = self.decision_function(X, return_Zhist=True)
        return self.model_Y.predict_proba(Zhist)
    
    def predict(self, X):
         yp = self.predict_proba(X)[:,1]
         return (yp>=self.best_yp_thres).astype(int)
    
    def save(self, path):
        assert path.endswith('.zip')
        folder = os.path.splitext(path)[0]
        os.makedirs(folder, exist_ok=True)
        self.model_Z.trainer.save_checkpoint(os.path.join(folder, 'model_Z.ckpt'))
        delattr(self, 'model_Z')
        with open(os.path.join(folder, 'other_than_model_Z.pickle'), 'wb') as ff:
            pickle.dump(self, ff)
        shutil.make_archive(folder, 'zip', folder)
        shutil.rmtree(folder)
    
    @classmethod
    def load(cls, path):
        assert path.endswith('.zip')
        folder = os.path.splitext(path)[0]
        with zipfile.ZipFile(path, 'r') as ff:
            ff.extractall(folder)
        with open(os.path.join(folder, 'other_than_model_Z.pickle'), 'rb') as ff:
            obj = pickle.load(ff)
        obj.model_Z = BernoulliHMM.load_from_checkpoint(os.path.join(folder, 'model_Z.ckpt'))
        return obj


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
            import pdb;pdb.set_trace()
            if np.all(res.func_vals[-1]<res.func_vals[:-1]):
                if os.path.exists('current_best.zip'):
                    os.remove('current_best.zip')
                os.rename('tmp.zip', 'current_best.zip')
        def loss(params):
            n_components = params[0]
            thres = params[1:]
            model_new = MyHMMClassifier(thres=thres, n_components=n_components, binary_col_mask=binary_col_mask, random_state=random_state, verbose=False, n_jobs=n_jobs, warm_start_model_folder=warm_start_model_folder)
            model_new.fit(Xtr, ytr)
            import pdb;pdb.set_trace()
            model_new.save('tmp.zip')
            return -model_new.model_Y_best_cv_score_
        opt_res = gp_minimize(loss,
            [Integer(3,15)]+[Real(thres_bounds[0,ii],thres_bounds[1,ii]) for ii in range(thres_bounds.shape[1])],
            n_calls=100, n_initial_points=20, initial_point_generator='lhs', callback=mycallback,
            random_state=random_state, verbose=True, n_jobs=1)
        import pdb;pdb.set_trace()
        model_new = MyHMMClassifier.load('current_best.zip')
        if os.path.exists('current_best.zip'): os.remove('current_best.zip')
        if os.path.exists('tmp.zip'): os.remove('tmp.zip')
        models_new_cv.append(model_new.best_estimator_)
        print(f'best_scores(HMM) = {model_new.best_score_}')
        print(f'new model coef = {model_aasm.best_estimator_.model_Y.coef_}')
        
        Xte = [X[i] for i in teids]
        ypte_new = model_new.predict_proba(Xte)[:,1]
        
        ## train AASM model
        
        model_aasm = LogisticRegression(penalty='elasticnet', tol=0.0001, class_weight=class_weight, random_state=random_state, solver='saga', max_iter=10000)#, verbose=True)
        model_aasm = BayesSearchCV(model_aasm, {
                'C': (1e-3, 1e+3, 'log-uniform'),
                'l1_ratio': (0.01, 0.99, 'uniform'),
            },
            scoring='roc_auc', cv=Ncv, n_points=10, n_iter=50,
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
    
    with open(f'results_{outcome}.pickle', 'wb') as ff:
        pickle.dump({
        'sids_te':sids_te, 'yte':ytes, 'ypte_new':yptes_new, 'ypte_aasm':yptes_aasm,
        'Zp':Zp,
        'models_aasm_cv':models_aasm_cv, 'models_new_cv':models_new_cv,
        'model_new_final':model_new_final, 'model_aasm_final':model_aasm_final
        }, ff)

        
if __name__=='__main__':
    main()
    
