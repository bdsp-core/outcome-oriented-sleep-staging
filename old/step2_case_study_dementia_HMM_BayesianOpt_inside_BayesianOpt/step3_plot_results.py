import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
import seaborn
seaborn.set_style('ticks')
from sklearn.metrics import roc_auc_score, roc_curve

outcome = 'Dementia'

with open(f'results_{outcome}.pickle', 'rb') as ff:
    res = pickle.load(ff)
yte = np.concatenate(res['yte'])
ypte_new = np.concatenate(res['ypte_new'])
ypte_aasm = np.concatenate(res['ypte_aasm'])
Z = res['Zp']
Z = [np.argmax(z,axis=1) for z in Z]

epoch_time = 30
with open(f'dataset_{outcome}_epoch{epoch_time}s.pickle', 'rb') as ff:
    res = pickle.load(ff)
sids = res['sids']
X = res['X']
S = res['S']
Y = res['Y']
Xnames = res['Xnames']

N = len(Z)
X = X[:N]
S = S[:N]
Y = Y[:N]

X2 = np.concatenate(X, axis=0)
Z2 = np.concatenate(Z, axis=0)
nc = len(np.unique(Z2))

plt.close()

for Xnames_ in [['alpha_rel_bp_F', 'alpha_rel_bp_C', 'alpha_rel_bp_O', 'theta_rel_bp_F', 'theta_rel_bp_C', 'theta_rel_bp_O'], ['has_spindle_F', 'has_spindle_C', 'has_spindle_O'], ['sw_amp_ptp_F', 'sw_amp_ptp_C', 'sw_amp_ptp_O'], ['has_rem', 'emg_env_rank_mean']]:
    fig = plt.figure(figsize=(10,len(Xnames_)*1.5))

    for xi, xn in enumerate(Xnames_):
        ax = fig.add_subplot(len(Xnames_),1,xi+1)
        xi2 = Xnames.index(xn)
        ax.boxplot([X2[Z2==nci,xi2] for nci in range(nc)], positions=np.arange(nc)+1)
        ax.set_ylabel(xn, rotation=0, ha='right', va='center')
        if xi==len(Xnames_)-1:
            ax.set_xlabel('Z')
        else:
            ax.set_xticklabels([])
        ax.set_xticks(np.arange(nc)+1)
        seaborn.despine()

    plt.tight_layout()
plt.show()
import pdb;pdb.set_trace()

"""
colors = {'New':'k', 'AASM':'b'}

plt.close()
fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111)
for type_ in ['New', 'AASM']:
    ypte = eval('ypte_'+type_.lower())
    auc = roc_auc_score(yte, ypte)
    fpr, tpr, tt = roc_curve(yte, ypte)
    ax.plot(fpr, tpr, c=colors[type_], label=f'{type_}: AUC = {auc:.3f}')
ax.plot([0,1],[0,1], c='r', ls='--')
ax.legend(frameon=False, loc='lower right')
ax.grid(True)
ax.set_xlabel('FPR or 1-specificity')
ax.set_ylabel('TPR or sensitivity')
ax.set_xlim(-0.01,1.01)
ax.set_ylim(-0.01,1.01)
seaborn.despine()

plt.tight_layout()
#plt.show()
plt.savefig(f'AUC_{outcome}.png')
"""
