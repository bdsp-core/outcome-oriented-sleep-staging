import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, matthews_corrcoef
from skopt import BayesSearchCV
#from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


def do_clustering(X, nc, random_state=None):
    #model = BayesianGaussianMixture(
    #    n_components=nc, covariance_type='full',
    #    max_iter=100, n_init=3, init_params='k-means++',
    #    weight_concentration_prior_type='dirichlet_process',
    #    weight_concentration_prior=None,
    #    random_state=random_state, verbose=10)
    model = GaussianMixture(
        n_components=nc, covariance_type='full',
        max_iter=100, n_init=3, init_params='k-means++',
        random_state=random_state,
        verbose=10, verbose_interval=5)
    model.fit(X)
    cluster = model.predict(X)
    print(model.weights_)

    # remove minor clusters
    bad_clusters = np.where(model.weights_<=0.05)[0]
    good_clusters = np.where(model.weights_>0.05)[0]
    good_ids = np.in1d(cluster, good_clusters)
    Xgood = X[good_ids]; ygood = cluster[good_ids]
    nnb = NearestNeighbors(n_neighbors=1).fit(Xgood)
    for cl in bad_clusters:
        ids = cluster==cl
        if ids.sum()==0:
            continue
        _, dist_ids = nnb.kneighbors(X[ids])
        cluster[ids] = ygood[dist_ids[:,0]]

    # re-order cluster to 0,1,2... according to size in descending order
    unique_clusters = np.array(list(set(cluster)))
    cc = [(cluster==x).sum() for x in unique_clusters]
    unique_clusters = unique_clusters[np.argsort(cc)[::-1]] # descending order
    cluster_mapping = {v:k for k,v in enumerate(unique_clusters)}
    cluster = np.vectorize(cluster_mapping.get)(cluster)
    return cluster, model


def do_classification(X, y, cvids, random_state=None, n_jobs=1, verbose=True):
    yp_cv = np.zeros(len(X))+np.nan
    hparams_cv = []
    models_cv = []
    model_name = 'logreg'#TODO

    cvs = sorted(set(cvids))
    for cvi in tqdm(cvs+['final'], disable=not verbose):
        if cvi=='final':  # final fit
            Xtr = X; ytr = y
        else:
            trids = cvids!=cvi
            Xtr = X[trids]; ytr = y[trids]


        # define model
        if model_name=='logreg':
            if cvi=='final':
                rs = random_state
            else:
                rs = random_state+(cvi+1)*2
            model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced',
                random_state=rs, max_iter=1000)
            hparams = {
                'C': (1e-3, 1e+3, 'log-uniform'),
                #'l1_ratio': (0.01, 0.99, 'uniform'),
                }
            scorer = 'roc_auc'
        else:
            raise NotImplementedError(f'Unknown model name {model_name}')
        if cvi=='final':  # final fit
            for pn in hparams_cv[0]:
                vals = [hp[pn] for hp in hparams_cv]
                is_int = all([type(x)==int for x in vals])
                val = np.median(vals)
                if is_int:
                    val = int(val)
                model.set_params(**{pn:val})
        else:
            # hparam search
            model = BayesSearchCV(model, hparams,
                n_iter=20, scoring=scorer, n_jobs=n_jobs, n_points=8,
                cv=10, random_state=random_state+(cvi+1)*2+1,
                verbose=0)

        # fit
        model.fit(Xtr, ytr)
        if cvi!='final':
            hparams_cv.append(model.best_params_)
            model = model.best_estimator_

        # calibration
        model = CalibratedClassifierCV(model, cv='prefit')
        model.fit(Xtr, ytr)

        if cvi=='final':  # final fit
            yp_final = model.predict_proba(Xtr)[:,1]
        else:
            models_cv.append(model)
            Xte = X[cvids==cvi]
            yp_cv[cvids==cvi] = model.predict_proba(Xte)[:,1]

    return models_cv, model, yp_cv, yp_final


def get_classification_perf(y, yp, nbt=1000, verbose=True, random_state=None):
    """
    performance with CI
    """
    if random_state is None:
        random_state = np.random.randn(0,10000)
    np.random.seed(random_state)

    y = np.array(y).astype(int)
    assert set(y)==set([0,1]), 'y must be binary, 0 or 1'
    aurocs = []
    auprcs = []
    fpr_curves = []
    tpr_curves = []
    tt_roc_curves = []
    pre_curves = []
    rec_curves = []
    tt_prc_curves = []
    op_point_methods = ['min-distance-to-perfect', 'Youden', 'sens80', 'sens90', 'spec80', 'spec90']
    accs = {x:[] for x in op_point_methods}
    f1s = {x:[] for x in op_point_methods}
    mccs = {x:[] for x in op_point_methods}
    sens = {x:[] for x in op_point_methods}
    spes = {x:[] for x in op_point_methods}
    ppvs = {x:[] for x in op_point_methods}
    npvs = {x:[] for x in op_point_methods}
    cms = {}
    for bti in tqdm(range(nbt+1), disable=not verbose):
        try:
            if bti==0:
                ybt = y
                ypbt = yp
            else:
                btids = np.random.choice(len(y),len(y),replace=True)
                ybt = y[btids]
                ypbt = yp[btids]

            aurocs.append( roc_auc_score(ybt, ypbt) )
            auprcs.append( average_precision_score(ybt, ypbt) )

            fpr, tpr, tt1 = roc_curve(ybt, ypbt)
            fpr_curves.append(fpr); tpr_curves.append(tpr); tt_roc_curves.append(tt1)

            pre, rec, tt2 = precision_recall_curve(ybt, ypbt)
            pre_curves.append(pre); rec_curves.append(rec); tt_prc_curves.append(tt2)

            if bti==0:
                thres = {
                    'min-distance-to-perfect': tt1[np.argmin(fpr**2+(1-tpr)**2)],
                    'Youden': tt1[np.argmax(tpr-fpr)],
                    'sens80': tt1[min([i for i, x in enumerate(tpr) if x>0.8], key=lambda i: tpr[i]-0.8)],
                    'sens90': tt1[min([i for i, x in enumerate(tpr) if x>0.9], key=lambda i: tpr[i]-0.9)],
                    'spec80': tt1[min([i for i, x in enumerate(1-fpr) if x>0.8], key=lambda i: 1-fpr[i]-0.8)],
                    'spec90': tt1[min([i for i, x in enumerate(1-fpr) if x>0.9], key=lambda i: 1-fpr[i]-0.9)],
                    }
            for op_point in op_point_methods:
                ypb = (ypbt>=thres[op_point]).astype(int)
                tp = np.sum((ybt==1)&(ypb==1))
                fn = np.sum((ybt==1)&(ypb==0))
                fp = np.sum((ybt==0)&(ypb==1))
                tn = np.sum((ybt==0)&(ypb==0))
                p = tp+fn; n = fp+tn
                pp = tp+fp; pn = fn+tn
                N = p+n
                if bti==0:
                    cms[op_point] = np.array([[tn,fp],[fn,tp]])#confusion_matrix(ybt, ypb)
                accs[op_point].append( (tp+tn)/N )
                f1s[op_point].append( 2*tp/(2*tp+fp+fn) )#f1_score(ybt, ypb)
                mccs[op_point].append( matthews_corrcoef(ybt, ypb) )
                sens[op_point].append( tp/p )#tpr[op_point]
                spes[op_point].append( tn/n )#1-fpr[op_point]
                ppvs[op_point].append( tp/pp )#precision_score(ybt,ypb)
                npvs[op_point].append( tn/pn )
        except Exception as ee:
            continue
    
    index = ['AUROC', 'AUPRC']
    perf_data = [
        [aurocs[0], np.percentile(aurocs[1:], 2.5), np.percentile(aurocs[1:], 97.5)],
        [auprcs[0], np.percentile(auprcs[1:], 2.5), np.percentile(auprcs[1:], 97.5)], ]
    for op_point in op_point_methods:
        for m, mn in zip(['accs', 'f1s', 'mccs', 'sens', 'spes', 'ppvs', 'npvs'], ['Accuracy', 'F1', 'MCC', 'Sensitivity', 'Specificity', 'PPV', 'NPV']):
            x = eval(f'{m}[op_point]')
            perf_data.append( [x[0], np.percentile(x[1:], 2.5), np.percentile(x[1:], 97.5)] )
            index.append(f'{op_point}:{mn}')
    perf = pd.DataFrame(data=np.array(perf_data), columns=['Val', 'LB', 'UB'], index=index)

    fpr_bt = np.sort(np.unique(np.concatenate(fpr_curves[1:]).round(3)))
    tpr_curves_bt = []
    for i in range(1,len(fpr_curves)):
        f = interp1d(fpr_curves[i], tpr_curves[i], kind='linear', bounds_error=False)
        tpr_curves_bt.append(f(fpr_bt))
    tpr_bt_ci = np.nanpercentile(np.array(tpr_curves_bt), (2.5, 97.5), axis=0)

    pre_bt = np.sort(np.unique(np.concatenate(pre_curves[1:]).round(3)))
    rec_curves_bt = []
    for i in range(1,len(pre_curves)):
        f = interp1d(pre_curves[i], rec_curves[i], kind='linear', bounds_error=False)
        rec_curves_bt.append(f(pre_bt))
    rec_bt_ci = np.nanpercentile(np.array(rec_curves_bt), (2.5, 97.5), axis=0)

    return (perf, cms,
            fpr_curves[0], tpr_curves[0], tt_roc_curves[0], fpr_bt, tpr_bt_ci,
            pre_curves[0], rec_curves[0], tt_prc_curves[0], pre_bt, rec_bt_ci)


def plot_vis(Xvis, X, Xnames, cluster, cluster_weight, path=None):
    Ks = sorted(set(cluster))
    cmap = matplotlib.colormaps['tab10']
    colors = {i:cmap(i/10) for i in range(10)}

    plt.close()
    fig = plt.figure(figsize=(14,8))
    gs = fig.add_gridspec(len(Ks), 2)

    # scatter plot
    ax = fig.add_subplot(gs[:,0])
    for k in Ks:
        ax.scatter(Xvis[cluster==k][:,0], Xvis[cluster==k][:,1], fc=colors[k], ec='none', s=5)#, alpha=0.3)

    # for legend only
    xx = Xvis[:,0].max()+1
    yy = Xvis[:,1].mean()
    for k in Ks:
        ax.scatter([xx], [yy], fc=colors[k], ec='none', s=30, label=f'Cluster {int(k)+1} ({cluster_weight[k]*100:.0f}%)')#, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_xlim(Xvis[:,0].min()-0.001, Xvis[:,0].max()+0.001)

    ax.axis('off')

    # feature boxplots
    for ki, k in enumerate(Ks):
        xs = [X[cluster==k][:,i] for i in range(X.shape[1])]
        if ki==0:
            ax = fig.add_subplot(gs[ki,1])
            ax0 = ax
        else:
            ax = fig.add_subplot(gs[ki,1], sharex=ax0, sharey=ax0)
        ax.boxplot(xs, positions=np.arange(X.shape[1]), showfliers=False)
        ax.text(0.01, 1, f'Cluster {int(k)+1}', ha='left', va='top', color=colors[k], transform=ax.transAxes)
        ax.set_xticks(np.arange(X.shape[1]))
        ax.set_xticklabels(Xnames, rotation=-30, ha='left')
        ax.set_ylabel('z scale')
        if ki<len(Ks)-1:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.yaxis.grid(True)
        sns.despine()

    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def main():
    epoch_time = int(sys.argv[1])
    n_state = int(sys.argv[2])
    random_state = 2023
    n_jobs = 8
    result_folder = f'results_{epoch_time}s_nc{n_state}'
    os.makedirs(result_folder, exist_ok=True)

    # get X
    df_feat = pd.read_csv(f'../data/features_epoch{epoch_time}s.csv.zip')
    unique_sids = df_feat.HashID.unique()
    sids = df_feat.HashID.values
    Xnames = list(df_feat.columns)
    Xnames.remove('HashID')
    Xnames.remove('DOVshifted')
    Xnames.remove('SleepStage')
    Xnames.remove('EpochStartIdx')
    X = df_feat[Xnames].values
    Xnames[-1] = 'emg_env'
    bin_mask = np.array([set(X[:,i])==set([0,1]) for i in range(X.shape[1])])

    # get S
    S = df_feat.SleepStage.values

    # remove nan is X and S
    ids = ~np.isnan(np.c_[X,S]).any(axis=1)
    sids = sids[ids]
    X = X[ids]
    S = S[ids]

    # get Y
    df_y = pd.read_csv(f'../data/mastersheet_matched_Dementia_alloutcomes.csv')
    assert (df_y.HashID==unique_sids).all()
    ycols = ['Y_Dementia',
        'Y_Hypertension',
        'Y_Depression',
        'Y_Atrial_Fibrillation',
        'Y_Myocardial_Infarction',]
    Y = df_y[ycols].values
    assert len(set(sids))==len(Y)

    # get CV ids
    dfcv = pd.read_csv(f'../step2_dementia_HMM_binary/cv_split_Dementia_N=318_seed{random_state}.csv')
    sid2cv = {dfcv.HashID.iloc[i]:int(dfcv.CV.iloc[i]) for i in range(len(dfcv))}

    # standardize
    Xmean = np.nanmean(X, axis=0)
    Xstd = np.nanstd(X, axis=0)
    Xmean[bin_mask] = 0
    Xstd[bin_mask] = 1
    X = (X-Xmean)/Xstd

    subsample_ratio = min(2,int(30/epoch_time))
    if subsample_ratio!=1:
        print(f'subsample by {subsample_ratio}')
        # subsample to make everything faster
        np.random.seed(random_state)
        ids = np.sort(np.random.choice(len(unique_sids),len(unique_sids)//subsample_ratio,replace=False))
        unique_sids = unique_sids[ids]
        Y = Y[ids]
        ids2 = np.in1d(sids, unique_sids)
        X = X[ids2]
        S = S[ids2]
        sids = sids[ids2]

    # reduce dimension
    pca = PCA(n_components=0.95, random_state=random_state).fit(X)
    X2 = pca.transform(X)

    print(f'Xnames = {Xnames}')
    print(f'N(subject) = {len(Y)}')
    print(f'N(epoch) = {len(X)}')

    # do clustering for each sleep stage
    sleep_stage_nums = [[1,2],[3],[4],[5]]
    sleep_stage_txts = ['N2+N3', 'N1', 'R', 'W']
    clusters = np.zeros(len(X))
    cluster_models = {}
    #vis_models = {}
    #Xvis = {}
    cluster_id_offset = 0
    for ss, ss_txt in zip(sleep_stage_nums, sleep_stage_txts):
        print(ss_txt)
        save_path = os.path.join(result_folder, f'cluster_model_{ss_txt}.pickle')
        if os.path.exists(save_path):
            print(f'Reading from {save_path}')
            with open(save_path,'rb') as ff:
                res = pickle.load(ff)
            #vis_models[ss_txt] = res['vis_model']
            #Xvis[ss_txt] = res['Xvis']
            cluster_models[ss_txt] = res['cluster_model']
            cluster = res['cluster']
            clusters[res['ids']] = cluster+cluster_id_offset
            cluster_id_offset += cluster.max()+1

        else:
            ids = np.in1d(S, ss)&(~np.any(np.isnan(X2), axis=1))
            sids_ = sids[ids]
            X_ = X2[ids]
            unique_sids_ = pd.unique(sids_)
            #y_ = pd.DataFrame(data={'HashID':unique_sids_}).merge(df_y, on='HashID', how='left', validate='1:1')[ycols].values

            cluster, cluster_models[ss_txt] = do_clustering(X_, n_state, random_state=random_state)
            print(f'{ss_txt}: N(cluster) = {len(set(cluster))}')
            clusters[ids] = cluster+cluster_id_offset
            cluster_id_offset += cluster.max()+1

            #vis_models[ss_txt] = UMAP(n_components=2, n_neighbors=100, min_dist=0.5, n_jobs=n_jobs, verbose=True).fit(X_, cluster)
            #Xvis[ss_txt] = vis_models[ss_txt].transform(X_)

            with open(save_path,'wb') as ff:
                pickle.dump({
                    #'vis_model':vis_models[ss_txt],
                    'cluster_model':cluster_models[ss_txt],
                    #'Xvis':Xvis[ss_txt],
                    'cluster':cluster,
                    'ids':np.where(ids)[0]
                    }, ff)

    # get correlation between subjects and clusters
    le = LabelEncoder().fit(sids)
    sids2 = le.transform(sids)
    cf = confusion_matrix(sids2, clusters)
    cf = cf[:,:len(set(clusters))]
    test_res = chi2_contingency(cf)
    print(f'Association between subject and cluster: F-test stat={test_res.statistic}, p={test_res.pvalue}, dof={test_res.dof}')
    # Association between subject and cluster: F-test stat=446543.1329547816, p=0.0, dof=5706

    """
    # plot
    for ss, ss_txt in zip(sleep_stage_nums, sleep_stage_txts):
        print(f'plotting {ss_txt}')
        ids = np.in1d(S, ss)&(~np.any(np.isnan(X), axis=1))
        sids_ = sids[ids]
        cluster = clusters[ids]
        cluster = cluster-cluster.min()
        uc = np.unique(cluster)
        cc = {k:np.mean(cluster==k) for k in uc}
        plot_vis(Xvis[ss_txt], X[ids], Xnames, cluster, cc,
                path=os.path.join(result_folder, f'vis_cluster_{ss_txt}.png'))
    """
    
    # get outcome classification performance
    states = sorted(set(clusters))
    Xhist = []
    for sid in unique_sids:
        cluster_ = clusters[sids==sid]
        Xhist.append( [np.sum(cluster_==s) for s in states] ) #TODO mean or sum or both?
    Xhist = np.array(Xhist)

    for yi, ycol in enumerate(ycols):
        print(ycol)
        goodids = ~np.isnan(Y[:,yi])
        X_ = Xhist[goodids]
        y_ = Y[:,yi][goodids]
        cv = np.array([sid2cv[x] for x in unique_sids[goodids]])
        model_cv, model, yp_cv, yp_final = do_classification(X_, y_, cv, n_jobs=n_jobs, random_state=random_state)
        perf_cv, cms, fpr, tpr, tt_roc, fpr_bt, tpr_ci, pre, rec, tt_prc, pre_bt, rec_ci = get_classification_perf(y_, yp_cv, random_state=random_state, nbt=100)##
        print(perf_cv)
        perf_cv.to_excel(os.path.join(result_folder, f'perf_cv_{ycol}.xlsx'))
        print(cms)
        #TODO save and plot confusion matrix
        #TODO save and plot ROC
        #TODO save and plot PRC

        with open(os.path.join(result_folder, f'clf_model_{ycol}.pickle'),'wb') as ff:
            pickle.dump({
                'outcome_clf_model':model,
                'outcome_clf_models_cv':model_cv,
                'confusion_matrix':cms,
                'fpr':fpr, 'tpr':tpr, 'tt_roc':tt_roc, 'fpr_bt':fpr_bt, 'tpr_ci':tpr_ci,
                'pre':pre, 'rec':rec, 'tt_prc':tt_prc, 'pre_bt':pre_bt, 'rec_ci':rec_ci,
                }, ff)



if __name__=='__main__':
    main()
    
