import sys
import os
import pickle
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
import torch as th
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, Subset
from models import HMMOOSSClassifier, th2np, MyDataset
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


def main():
    outcome = 'Dementia'
    epoch_time = int(sys.argv[1])#30
    random_state = 2023

    figure_folder = f'figures_epochtime{epoch_time}s'
    os.makedirs(figure_folder, exist_ok=True)

    df_feat = pd.read_csv(f'features_epoch{epoch_time}s.csv.zip')
    sids = df_feat.HashID.values
    unique_sids = pd.unique(sids)
    S = df_feat.SleepStage.values
    #S = [S[sids==x] for x in unique_sids]
    Xnames = list(df_feat.columns)
    Xnames.remove('HashID')
    Xnames.remove('DOVshifted')
    Xnames.remove('SleepStage')
    Xnames.remove('EpochStartIdx')
    X = df_feat[Xnames].values
    X = [X[sids==x] for x in unique_sids]
    df_y = pd.read_csv(f'../data/mastersheet_matched_{outcome}.csv')
    Y = df_y[f'Y_{outcome}'].values.astype(int)
    N = len(Y)

    with open(os.path.join(f'results_new_{outcome}_epoch{epoch_time}s', 'results.pickle'), 'rb') as ff:
        res = pickle.load(ff)
    Z = np.concatenate([np.argmax(z,axis=1) for z in res['Zp']])
    n_state = res['Zp'][0].shape[1]
        
    model_path = os.path.join(f'results_new_{outcome}_epoch{epoch_time}s', 'model_final.ckpt')
        
    #TODO load from model
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
    weights = th.load(model_path, map_location=th.device('cpu'))['state_dict']
    model.unnormalized_thres = weights['unnormalized_thres']
    model.unnormalized_transition_matrix = weights['unnormalized_transition_matrix']
    model.unnormalized_emission_matrix = weights['unnormalized_emission_matrix']
    model.unnormalized_state_priors = weights['unnormalized_state_priors']
    model.coef_th_ = weights['coef_th_']
    model.intercept_th_ = weights['intercept_th_']
    
    model.n_components = model.unnormalized_transition_matrix.shape[1]
    model.Xmean_ = np.mean(X2, axis=0)
    model.Xscale_ = np.std(X2, axis=0)
    #model.mytrainer = model._get_trainer()
    
    model.X_thres_ = sigmoid(th2np(model.unnormalized_thres))*(thres_bounds[1]-thres_bounds[0])+thres_bounds[0]
    model.startprob_ = th2np(th.softmax(F.pad(model.unnormalized_state_priors, (1,0), "constant", 0), dim=0))
    model.transmat_ = th2np(th.softmax(F.pad(model.unnormalized_transition_matrix, (0,0,1,0), "constant", 0), dim=0)).T
    model.emissionprob_ = th2np(th.sigmoid(model.unnormalized_emission_matrix))
    model.coef_ = th2np(model.coef_th_)
    model.intercept_ = th2np(model.intercept_th_)
    model.thres_bounds = th.tensor(model.thres_bounds).float()

    df_X = pd.DataFrame(data={'Name':Xnames, 'threshold':model.X_thres_})
    print(df_X)

    print(model.coef_)

    ids = (~np.isnan(Z))&(~np.isnan(S))
    Z = Z[ids]
    S = S[ids]
    cm = confusion_matrix((S-1).astype(int), Z)
    cm = cm[:5]
    cm = cm/cm.sum(axis=1, keepdims=True)

    plt.close()
    fig = plt.figure(figsize=(6,4))

    ax = fig.add_subplot(111)
    sns.heatmap(cm,
            vmin=0, vmax=1, cmap='Blues',
            annot=True, fmt='.2f', cbar=False, square=True,
            linewidths=0.5, linecolor='w',
            xticklabels=[str(x+1) for x in range(n_state)],
            yticklabels=['N3', 'N2', 'N1', 'R', 'W'],
            )
    ax.set(ylabel="", xlabel="New State")
    ax.xaxis.tick_top()
    plt.yticks(rotation=0)
    #plt.xticks(rotation=40, ha='left')

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(figure_folder, f'AASM2new_epochtime{epoch_time}s.png'), bbox_inches='tight', dpi=300)


    cm = confusion_matrix(Z, (S-1).astype(int))
    cm = cm[:,:5]
    cm = cm/cm.sum(axis=1, keepdims=True)

    plt.close()
    fig = plt.figure(figsize=(4,6))

    ax = fig.add_subplot(111)
    sns.heatmap(cm,
            vmin=0, vmax=1, cmap='Blues',
            annot=True, fmt='.2f', cbar=False, square=True,
            linewidths=0.5, linecolor='w',
            xticklabels=['N3', 'N2', 'N1', 'R', 'W'],
            yticklabels=[str(x+1) for x in range(n_state)]
            )
    ax.set(xlabel="", ylabel="New State")
    ax.xaxis.tick_top()
    plt.yticks(rotation=0)
    #plt.xticks(rotation=40, ha='left')

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(figure_folder, f'new2AASM_epochtime{epoch_time}s.png'), bbox_inches='tight', dpi=300)

    
    plt.close()
    fig = plt.figure(figsize=(4,4))

    ax = fig.add_subplot(111)
    sns.heatmap(model.transmat_,
            vmin=0, vmax=1, cmap='Blues',
            annot=True, fmt='.2f', cbar=False, square=True,
            linewidths=0.5, linecolor='w',
            xticklabels=[str(x+1) for x in range(n_state)],
            yticklabels=[str(x+1) for x in range(n_state)]
            )
    ax.set(xlabel="New State", ylabel="New State")
    ax.xaxis.tick_top()
    plt.yticks(rotation=0)
    #plt.xticks(rotation=40, ha='left')

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(figure_folder, f'transition_epochtime{epoch_time}s.png'), bbox_inches='tight', dpi=300)

    plt.close()
    fig = plt.figure(figsize=(9,6))

    ax = fig.add_subplot(111)
    sns.heatmap(model.emissionprob_,
            vmin=0, vmax=1, cmap='Blues',
            annot=True, fmt='.2f', cbar=False, square=True,
            linewidths=0.5, linecolor='w',
            xticklabels=Xnames, 
            yticklabels=[str(x+1) for x in range(n_state)]
            )
    ax.set(xlabel="", ylabel="New State")
    ax.xaxis.tick_top()
    plt.yticks(rotation=0)
    plt.xticks(rotation=40, ha='left')

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(figure_folder, f'emission_epochtime{epoch_time}s.png'), bbox_inches='tight', dpi=300)
    
    
if __name__=='__main__':
    main()
