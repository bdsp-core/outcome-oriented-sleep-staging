import numpy as np
import pandas as pd
from pacmap import PaCMAP
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')

random_state = 2023
np.random.seed(random_state)

#method = 'UMAP'
method = 'PaCMAP'

epoch_times = [30, 15, 10, 5]
Xnames = ['alpha_rel_bp_F','alpha_rel_bp_C', 'alpha_rel_bp_O', 'theta_rel_bp_F', 'theta_rel_bp_C', 'theta_rel_bp_O', 'has_spindle_F', 'has_spindle_C', 'has_spindle_O', 'sw_amp_ptp_F', 'sw_amp_ptp_C', 'sw_amp_ptp_O', 'sw_perc_F', 'sw_perc_C', 'sw_perc_O', 'has_rem', 'emg_env_rank_mean']

stages_num = [1,2,3,4,5]
stage_num2color = {5:'k', 4:'r', 3:'y', 2:'g', 1:'c'}
stage_num2txt   = {5:'W', 4:'R', 3:'N1',2:'N2',1:'N3'}

for epoch_time in epoch_times:
    df = pd.read_csv(f'features_epoch{epoch_time}s_tmp.csv')

    # subsample
    ids = np.sort(np.random.choice(len(df), len(df)//10, replace=False))
    df = df.iloc[ids].reset_index(drop=True)

    X = df[Xnames].values
    #assert np.all(pd.notna(X))
    X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    y = df.SleepStage.values

    goodids = ~np.any(np.isnan(np.c_[X,y]), axis=1)
    X = X[goodids]
    y = y[goodids]

    if method=='PaCMAP':
        vis = PaCMAP(save_tree=True, random_state=random_state)
    elif method=='UMAP':
        vis = UMAP()
    vis.fit(X)
    Xvis = vis.transform(X)

    lim = [Xvis[:,0].min()-1, Xvis[:,0].max()+1]
    figsize = (7,7)
    plt.close()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for stage in stages_num:
        ax.scatter(Xvis[y==stage,0], Xvis[y==stage,1], fc=stage_num2color[stage], ec='none', s=30, alpha=0.1)
    for stage in stages_num:
        ax.scatter([lim[1]+10],lim[1],fc=stage_num2color[stage], ec='none',s=30, label=stage_num2txt[stage])
    ax.set_xlabel(f'{method} dim 1')
    ax.set_ylabel(f'{method} dim 2')
    ax.set_xlim(lim)
    #ax.set_ylim(lim)
    ax.legend()
    seaborn.despine()

    plt.tight_layout()
    #plt.show()
    plt.savefig(f'vis_{method}_epochtime{epoch_time}s.png')

