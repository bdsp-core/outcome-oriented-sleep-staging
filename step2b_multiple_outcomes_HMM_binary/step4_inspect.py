import sys
import os
import pickle
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
import torch as th
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, Subset
from models import HMMOOSSClassifier, th2np, MyDataset
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


def main():
    epoch_time = int(sys.argv[1])#30
    random_state = 2023

    figure_folder = f'figures_epochtime{epoch_time}s'
    os.makedirs(figure_folder, exist_ok=True)

    df_feat = pd.read_csv(f'../data/features_epoch{epoch_time}s.csv.zip')
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
    df_y = pd.read_csv('../data/mastersheet_matched_Dementia_alloutcomes.csv')
    #Y = df_y[f'Y_{outcome}'].values.astype(int)
    N = len(df_y)

    with open(os.path.join(f'results_new_multiple_outcomes_epoch{epoch_time}s', 'results.pickle'), 'rb') as ff:
        res = pickle.load(ff)
    Zp = np.concatenate([z for z in res['Zp']], axis=0)
    Z = np.argmax(Zp, axis=1)
    n_state = Zp.shape[1]
    print(f'n_state = {n_state}')

    outcomes = res['outcomes']
    yte = res['yte']
    ypte = res['ypte']
    for yi, ycol in enumerate(outcomes):
        ids = ~np.isnan(yte[:,yi])
        auc = roc_auc_score(yte[ids][:,yi], ypte[ids][:,yi])
        print(f'{ycol}:  CV AUC new = {auc}')
        
    model_path = os.path.join(f'results_new_multiple_outcomes_epoch{epoch_time}s', 'model_final.ckpt')
        
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

    ids = (~np.isnan(Z))&(~np.isnan(S))
    Z = Z[ids]
    S = S[ids]
    Z2 = []; S2 = []
    for i in range(n_state):
        if (Z==i).sum()==0:
            Z2.append(i)
            S2.append(5)
    Z = np.r_[Z, Z2]
    S = np.r_[S, S2]

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


    bound = np.abs(model.coef_).max()
    assert bound<3
    bound = 3
    outcome2txt = {'Y_Dementia':'Dementia', 'Y_Hypertension':'HTN', 'Y_Depression':'Depression', 'Y_Atrial_Fibrillation':'AFib', 'Y_Myocardial_Infarction':'MI'}

    #TODO
    ids = [0,1, 3,4,5]
    coef_ = model.coef_[:,ids]
    xticklabels = [outcome2txt[outcomes[x]] for x in ids]

    print(coef_)

    plt.close()
    fig = plt.figure(figsize=(coef_.shape[1]*0.6, coef_.shape[0]*0.6))

    ax = fig.add_subplot(111)
    sns.heatmap(coef_,
            vmin=-bound, vmax=bound, cmap='coolwarm',
            annot=True, fmt='.2f', cbar=False, square=True,
            linewidths=0.5, linecolor='w',
            xticklabels=xticklabels,
            yticklabels=[str(x+1) for x in range(n_state-1)]
            )
    ax.set(ylabel="New State")
    ax.xaxis.tick_top()
    plt.yticks(rotation=0)
    plt.xticks(rotation=40, ha='left')

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(figure_folder, f'outcome_coef_{epoch_time}s.png'), bbox_inches='tight', dpi=300)
    
    
if __name__=='__main__':
    main()
