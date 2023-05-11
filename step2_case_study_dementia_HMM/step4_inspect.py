import pickle
import numpy as np
from scipy.special import expit as sigmoid
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from models2 import HMMOOSSClassifier, th2np, MyDataset


def main():
    outcome = 'Dementia'
    epoch_time = 30
    random_state = 2023
    
    with open(f'dataset_{outcome}_epoch{epoch_time}s.pickle', 'rb') as ff:
        res = pickle.load(ff)
    sids = res['sids']
    X = res['X']
    Y = res['Y']
    S = res['S']
    Xnames = res['Xnames']
    N = len(Y)
    print(f'N = {N}')
    print(f'Xnames = {Xnames}')
        
    model_path = 'lightning_logs/version_56/checkpoints/epoch=22-step=667.ckpt'
        
    X2 = np.concatenate(X, axis=0)
    D = X2.shape[1]
    binary_col_mask = np.array([set(X2[:,i])==set([0,1]) for i in range(D)])
    X3 = X2[:,~binary_col_mask]
    X3[X3==0] = np.nan
    thres_bounds = np.zeros((2,D))
    thres_bounds[:,~binary_col_mask] = np.nanpercentile(X3, (5,95), axis=0)
    thres_bounds[0,binary_col_mask] = 0.49
    thres_bounds[1,binary_col_mask] = 0.51
    
    model = HMMOOSSClassifier(thres_bounds, D, Xnames=Xnames, random_state=random_state, verbose=True)
    weights = th.load(model_path)['state_dict']
    model.unnormalized_thres = weights['unnormalized_thres']
    model.unnormalized_transition_matrix = weights['unnormalized_transition_matrix']
    model.unnormalized_emission_matrix = weights['unnormalized_emission_matrix']
    model.unnormalized_state_priors = weights['unnormalized_state_priors']
    model.coef_th_ = weights['coef_th_']
    model.intercept_th_ = weights['intercept_th_']
    
    model.n_components = model.unnormalized_transition_matrix.shape[1]
    model.Xmean_ = np.mean(X2, axis=0)
    model.Xscale_ = np.std(X2, axis=0)
    model.mytrainer = model._get_trainer()
    
    model.X_thres_ = sigmoid(th2np(model.unnormalized_thres))*(thres_bounds[1]-thres_bounds[0])+thres_bounds[0]
    model.startprob_ = th2np(th.softmax(F.pad(model.unnormalized_state_priors, (1,0), "constant", 0), dim=0))
    model.transmat_ = th2np(th.softmax(F.pad(model.unnormalized_transition_matrix, (0,0,1,0), "constant", 0), dim=0)).T
    model.emissionprob_ = th2np(th.sigmoid(model.unnormalized_emission_matrix))
    model.coef_ = th2np(model.coef_th_)
    model.intercept_ = th2np(model.intercept_th_)
    model.thres_bounds = th.tensor(model.thres_bounds).float()
    
    #zp = model.predict_proba_Z(X)
    X = [(x-model.Xmean_)/model.Xscale_ for x in X]
    dataset = MyDataset(X)
    loader = DataLoader(dataset, batch_size=model.batch_size, num_workers=0, shuffle=False, collate_fn=MyDataset.collate)
    outputs = model.mytrainer.predict(model=model, dataloaders=loader, ckpt_path=model_path)
    
    H = np.concatenate([x['H'] for x in outputs])
    zp = sum([x['zp'] for x in outputs], [])
    import pdb;pdb.set_trace()
    
if __name__=='__main__':
    main()
