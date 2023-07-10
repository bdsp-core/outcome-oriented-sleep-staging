import os
import pickle
import numpy as np
import pandas as pd
from pacmap import PaCMAP
from umap import UMAP
import matplotlib
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')

random_state = 2023
np.random.seed(random_state)

#method = 'UMAP'
method = 'PaCMAP'

outcome = 'Dementia'

epoch_times = [30, 15]#, 10, 5]
Xnames = ['alpha_rel_bp_F','alpha_rel_bp_C', 'alpha_rel_bp_O', 'theta_rel_bp_F', 'theta_rel_bp_C', 'theta_rel_bp_O', 'has_spindle_F', 'has_spindle_C', 'has_spindle_O', 'sw_amp_ptp_F', 'sw_amp_ptp_C', 'sw_amp_ptp_O', 'sw_perc_F', 'sw_perc_C', 'sw_perc_O', 'has_rem', 'emg_env_rank_mean']

stages_num = [1,2,3,4,5]
stage_num2color = {5:'k', 4:'r', 3:'y', 2:'g', 1:'c'}
stage_num2txt   = {5:'W', 4:'R', 3:'N1',2:'N2',1:'N3'}

for epoch_time in epoch_times:
    df = pd.read_csv(f'features_epoch{epoch_time}s.csv.zip')

    result_folder = f'results_new_{outcome}_epoch{epoch_time}s'
    with open(os.path.join(result_folder, f'results.pickle'), 'rb') as ff:
        res = pickle.load(ff)
    assert np.all(res['sids_te']==df.HashID)
    n_state = res['Zp'][0].shape[1]
    df['Z'] = np.argmax(np.concatenate(res['Zp'], axis=0), axis=1)

    # subsample
    ids = np.sort(np.random.choice(len(df), len(df)//10, replace=False))
    df = df.iloc[ids].reset_index(drop=True)

    X = df[Xnames].values
    #assert np.all(pd.notna(X))
    X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    y = df.SleepStage.values
    Z = df.Z.values

    goodids = ~np.any(np.isnan(np.c_[X,y,Z]), axis=1)
    X = X[goodids]
    y = y[goodids]
    Z = Z[goodids]

    if method=='PaCMAP':
        vis = PaCMAP(save_tree=True, random_state=random_state)
    elif method=='UMAP':
        vis = UMAP()
    vis.fit(X)
    Xvis = vis.transform(X)
    #x0 = (-5,-1.6)
    #dists2 = (Xvis[:,0]-x0[0])**2+(Xvis[:,1]-x0[1])**2
    #dists2 > 200

    xlim = [Xvis[:,0].min()-1, Xvis[:,0].max()+1]
    ylim = [Xvis[:,1].min()-1, Xvis[:,1].max()+1]
    figsize = (10.5,9)

    plt.close()
    fig = plt.figure(figsize=figsize)
    for si, stage in enumerate(stages_num):
        if si==0:
            ax = fig.add_subplot(2,3,stage); ax0 = ax
        else:
            ax = fig.add_subplot(2,3,stage, sharex=ax0, sharey=ax0)
        ax.scatter(Xvis[y==stage,0], Xvis[y==stage,1], fc=stage_num2color[stage], ec='none', s=30, alpha=0.1)
        ax.text(0.01, 0.99, stage_num2txt[stage], va='top', ha='left', transform=ax.transAxes)

    """
    ax = fig.add_subplot(2,3,6,sharex=ax0,sharey=ax0)
    for stage in stages_num:
        ax.scatter([xlim[1]+10],[ylim[1]+10],fc=stage_num2color[stage], ec='none',s=30, label=stage_num2txt[stage])
    ax.legend()
    """

    #ax.set_xlabel(f'{method} dim 1')
    #ax.set_ylabel(f'{method} dim 2')
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    seaborn.despine()

    plt.tight_layout()
    #plt.show()
    plt.savefig(f'vis_stages_{method}_epochtime{epoch_time}s.png')


    cmap = matplotlib.cm.get_cmap('turbo')
    figsize = (15,9)

    plt.close()
    fig = plt.figure(figsize=figsize)
    for z in range(n_state):
        if z==0:
            ax = fig.add_subplot(2,5,z+1); ax0 = ax
        else:
            ax = fig.add_subplot(2,5,z+1, sharex=ax0, sharey=ax0)
        ax.scatter(Xvis[Z==z,0], Xvis[Z==z,1], fc=cmap(z/(n_state-1)), ec='none', s=30, alpha=0.1)
        ax.text(0.01, 0.99, f'state={z+1}', va='top', ha='left', transform=ax.transAxes)

    #ax.set_xlabel(f'{method} dim 1')
    #ax.set_ylabel(f'{method} dim 2')
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    seaborn.despine()

    plt.tight_layout()
    #plt.show()
    plt.savefig(f'vis_Z_{method}_epochtime{epoch_time}s.png')
