import os
import shutil
import pickle
import zipfile
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from skopt import BayesSearchCV
from myhmmlearn import BernoulliHMM


class MyHMMClassifier(BaseEstimator, ClassifierMixin):
    """
    """
    def __init__(self, thres=None, binary_col_mask=None, n_components=2, random_state=None, verbose=False, n_jobs=1, hmm_n_iter=1, class_weight=None, warm_start_model_folder='warm_start_models'):
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
            self.model_Z.save(warm_start_path)
            print('finish')
        
        Z = self.model_Z.predict(Xb, T)
        Z = [Z[to_list_ids[i]:to_list_ids[i+1]] for i in range(len(to_list_ids)-1)]
        Zhist = np.array([[np.mean(z==zz) for zz in range(self.n_components-1)] for z in Z])

        # logistic regression
        Ncv = 5
        self.model_Y = LogisticRegression(penalty='elasticnet', tol=0.0001, class_weight=self.class_weight, random_state=self.random_state+100, solver='saga', max_iter=10000, verbose=False)
        self.model_Y = BayesSearchCV(self.model_Y,
               {'C': (1e-3, 1e+3, 'log-uniform'),
                'l1_ratio': (0.01, 0.99, 'uniform'),},
            scoring='roc_auc', cv=Ncv, n_points=1, n_iter=2,
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
        # path is either a directory or .zip
        is_zip = path.endswith('.zip')
        if is_zip:
            path = os.path.splitext(path)[0]
        os.makedirs(path, exist_ok=True)
        self.model_Z.save(os.path.join(path, 'model_Z.ckpt'))
        delattr(self, 'model_Z')
        with open(os.path.join(path, 'other_than_model_Z.pickle'), 'wb') as ff:
            pickle.dump(self, ff)
        if is_zip:
            shutil.make_archive(path, 'zip', path)
            shutil.rmtree(path)
    
    @classmethod
    def load(cls, path):
        # path is either a directory or .zip
        is_zip = path.endswith('.zip')
        if is_zip:
            folder = os.path.splitext(path)[0]
            with zipfile.ZipFile(path, 'r') as ff:
                ff.extractall(folder)
        else:
            folder = path
        with open(os.path.join(folder, 'other_than_model_Z.pickle'), 'rb') as ff:
            obj = pickle.load(ff)
        obj.model_Z = BernoulliHMM.load_from_checkpoint(os.path.join(folder, 'model_Z.ckpt'))
        if is_zip:
            shutil.rmtree(folder)
        return obj
