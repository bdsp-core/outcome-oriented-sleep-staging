import pickle
import numpy as np
from scipy.stats import mode
from hmmlearn.hmm import PoissonHMM
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from skopt import BayesSearchCV


class MyHMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, W, n_components=2, C=1, l1_ratio=0, random_state=None, verbose=False):
        self.W = W
        self.n_components = n_components
        self.C = C
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.verbose = verbose
        
    def fit(self, X, y):
        T = [len(x) for x in X]
        X2 = np.concatenate(X)
        ids = np.r_[0,np.cumsum(T)]
        X2 = np.round(X2*self.W).astype(int)
        self.model_Z = PoissonHMM(n_components=self.n_components, algorithm='viterbi', random_state=self.random_state, n_iter=1000, tol=0.001, verbose=self.verbose)
        self.model_Z.fit(X2, lengths=T)
        
        Z = self.model_Z.predict(X2, lengths=T)
        Z = [Z[ids[i]:ids[i+1]] for i in range(len(ids)-1)]
        Zhist = np.array([[np.mean(z==zz) for zz in range(self.n_components-1)] for z in Z])

        self.model_Y = LogisticRegression(penalty='elasticnet', tol=0.00001, C=self.C, class_weight='balanced', random_state=self.random_state, solver='saga', max_iter=1000, l1_ratio=self.l1_ratio)#, verbose=True)
        self.model_Y.fit(Zhist, y)
        return self
    
    #TODO def decision_function(self, X):
    def predict_proba(self, X, return_proba=True):
        T = [len(x) for x in X]
        X2 = np.concatenate(X)
        ids = np.r_[0,np.cumsum(T)]
        X2 = np.round(X2*self.W).astype(int)
        
        Z = self.model_Z.predict(X2, lengths=T)
        Z = [Z[ids[i]:ids[i+1]] for i in range(len(ids)-1)]
        Zhist = np.array([[np.mean(z==zz) for zz in range(self.n_components-1)] for z in Z])

        if return_proba:
            return self.model_Y.predict_proba(Zhist)
        else:
            return self.model_Y.predict(Zhist)
    
    def predict(self, X):
         return self.predict_proba(X, return_proba=False)
    

def main():
    outcome = 'Dementia'
    with open(f'../step2_case_study_dementia/dataset_{outcome}.pickle', 'rb') as ff:
        res = pickle.load(ff)
    sids = res['sids']
    X = res['X']
    S = res['S']
    Y = res['Y']
    Xnames = res['Xnames']
    random_state = 2023
    Ncv = 10
    W = 6000
    n_jobs = 8
    
    ncs = np.arange(2,6)
    teids_cv = []
    ytes = []
    yptes_new = []
    yptes_aasm = []
    models_Z_cv = []
    models_Y_aasm_cv = []
    models_Y_new_cv = []
    cvf = StratifiedKFold(n_splits=Ncv, shuffle=True, random_state=random_state)
    for cvi, (trids, teids) in enumerate(cvf.split(np.zeros((len(X),1)), Y)):
        teids_cv.append(teids)
        print(f'CV = {cvi}')
        Xtr = [X[i] for i in trids]
        Str = [S[i] for i in trids]
        ytr = Y[trids]
        
        T = [len(x) for x in Xtr]
        Xtr2 = np.concatenate(Xtr)
        ids = np.r_[0,np.cumsum(T)]
        Xtr2 = np.round(Xtr2*W).astype(int)
        best_scores = []
        model_Ys = []
        model_Zs = []
        for nc in ncs:
            print(nc)
            model_Z = PoissonHMM(n_components=nc, algorithm='viterbi', random_state=random_state, n_iter=10000, tol=0.01, verbose=False)
            model_Z.fit(Xtr2, lengths=T)
            model_Zs.append(model_Z)
            
            Z = model_Z.predict(Xtr2, lengths=T)
            Z = [Z[ids[i]:ids[i+1]] for i in range(len(ids)-1)]
            Zhist = np.array([[np.mean(z==zz) for zz in range(nc-1)] for z in Z])

            model_Y = LogisticRegression(penalty='elasticnet', tol=0.0001, class_weight='balanced', random_state=random_state, solver='saga', max_iter=10000)#, verbose=True)
            model_Y = BayesSearchCV(model_Y, {
                    'C': (1e-3, 1e+3, 'log-uniform'),
                    'l1_ratio': (0.01, 0.99, 'uniform'),
                    #'n_components': (2, 10),
                },
                scoring='roc_auc', cv=Ncv, n_points=n_jobs, n_iter=100,
                n_jobs=n_jobs, verbose=False, random_state=random_state,
                )
            model_Y.fit(Zhist, ytr)
            best_scores.append(model_Y.best_score_)
            model_Ys.append(model_Y.best_estimator_)
            print(best_scores[-1])
            print(model_Y.best_estimator_.coef_)
                  
        best_id = np.argmax(best_scores)
        model_Y = model_Ys[best_id]
        model_Z = model_Zs[best_id]
        best_score = best_scores[best_id]
        print(f'best_scores(HMM) = {best_score}')
        models_Z_cv.append(model_Z)
        models_Y_new_cv.append(model_Y)
        
        Xte = [X[i] for i in teids]
        yte = Y[teids]
        T = [len(x) for x in Xte]
        Xte2 = np.concatenate(Xte)
        ids = np.r_[0,np.cumsum(T)]
        Xte2 = np.round(Xte2*W).astype(int)
        Z = model_Z.predict(Xte2, lengths=T)
        Z = [Z[ids[i]:ids[i+1]] for i in range(len(ids)-1)]
        Zhist = np.array([[np.mean(z==zz) for zz in range(model_Z.n_components-1)] for z in Z])
        ypte_new = model_Y.predict_proba(Zhist)[:,1]
        
        model_Y = LogisticRegression(penalty='elasticnet', tol=0.0001, class_weight='balanced', random_state=random_state, solver='saga', max_iter=10000)#, verbose=True)
        model_Y = BayesSearchCV(model_Y, {
                'C': (1e-3, 1e+3, 'log-uniform'),
                'l1_ratio': (0.01, 0.99, 'uniform'),
                #'n_components': (2, 10),
            },
            scoring='roc_auc', cv=Ncv, n_points=n_jobs, n_iter=100,
            n_jobs=n_jobs, verbose=False, random_state=random_state,
            )
        Zhist = np.array([[np.mean(z==zz) for zz in range(1,5-1+1)] for z in Str])
        model_Y.fit(Zhist, ytr)
        print(f'best_scores(AASM) = {model_Y.best_score_}')
        print(model_Y.best_estimator_.coef_)
        models_Y_aasm_cv.append(model_Y.best_estimator_)
        
        Ste = [S[i] for i in teids]
        Zhist = np.array([[np.mean(z==zz) for zz in range(1,5-1+1)] for z in Ste])
        ypte_aasm = model_Y.predict_proba(Zhist)[:,1]
            
        ytes.append(yte)
        yptes_new.append(ypte_new)
        yptes_aasm.append(ypte_aasm)

    yte = np.concatenate(ytes)
    ypte_new = np.concatenate(yptes_new)
    ypte_aasm = np.concatenate(yptes_aasm)
    
    # fit final model
    nc = mode([m.n_components for m in models_Z_cv]).mode[0]
    T = [len(x) for x in X]
    X2 = np.concatenate(X)
    ids = np.r_[0,np.cumsum(T)]
    X2 = np.round(X2*W).astype(int)
    model_Z_final = PoissonHMM(n_components=nc, algorithm='viterbi', random_state=random_state, n_iter=10000, tol=0.01, verbose=False)
    model_Z_final.fit(X2, lengths=T)

    Z = model_Z_final.predict(X2, lengths=T)
    Z = [Z[ids[i]:ids[i+1]] for i in range(len(ids)-1)]
    Zhist = np.array([[np.mean(z==zz) for zz in range(nc-1)] for z in Z])
    model_Y_new_final = LogisticRegression(
        penalty='elasticnet', tol=0.0001, class_weight='balanced',
        random_state=random_state, solver='saga', max_iter=10000,
        C=np.median([m.C for m in models_Y_new_cv]),
        l1_ratio=np.median([m.l1_ratio for m in models_Y_new_cv]))
    Zhist = np.array([[np.mean(z==zz) for zz in range(1,5-1+1)] for z in S])
    model_Y_new_final.fit(Zhist, Y)
            
    model_Y_aasm_final = LogisticRegression(
        penalty='elasticnet', tol=0.0001, class_weight='balanced',
        random_state=random_state, solver='saga', max_iter=10000,
        C=np.median([m.C for m in models_Y_aasm_cv]),
        l1_ratio=np.median([m.l1_ratio for m in models_Y_aasm_cv]))
    Zhist = np.array([[np.mean(z==zz) for zz in range(1,5-1+1)] for z in S])
    model_Y_aasm_final.fit(Zhist, Y)
    
    with open('results.pickle', 'wb') as ff:
        pickle.dump({
        #'teids_cv':teids_cv,
        'yte':yte, 'ypte_new':ypte_new, 'ypte_aasm':ypte_aasm,
        'models_Z_cv':models_Z_cv, 'models_Y_aasm_cv':models_Y_aasm_cv, 'models_Y_new_cv':models_Y_new_cv,
        'model_Z_final':model_Z_final, 'model_Y_new_final':model_Y_new_final, 'model_Y_aasm_final':model_Y_aasm_final
        }, ff)

        
if __name__=='__main__':
    main()
    