from itertools import product, groupby
from collections import defaultdict
import datetime
import os
import pickle
import re
import subprocess
from io import StringIO
import numpy as np
import pandas as pd
from tqdm import tqdm
import mne
import h5py


def load_data(folder_path, ch_names=None, load_data=True, df_annot=None):
    # read annotations file
    if df_annot is None:  # read from dropbox
        annot_folder = r'dropbox_mgh:/BDSP_engineering/new_redacted_annotations'
        annot_path = os.path.join(annot_folder, folder_path, folder_path+'_annotations.csv')
        output = subprocess.check_output(['rclone', 'cat', annot_path]).decode()
        annot = pd.read_csv(StringIO(output))
    else:
        hashid, dov1, dov2 = folder_path.split('_')
        dov1 = datetime.datetime.strptime(dov1, '%Y%m%d')
        annot = df_annot[(df_annot.HashID==hashid)&(df_annot.DOVshifted==dov1)].reset_index(drop=True)

    # read signals file
    signal_folder = r'/sbgenomics/project-files/bdsp-opendata-repository/PSG/data/S0001'
    signal_path = os.path.join(signal_folder, folder_path, 'Shifted_Signal_'+folder_path+'.mat')
    with h5py.File(signal_path, 'r') as ff:
        Fs = ff['recording']['samplingrate'][()].item()
        start_time = datetime.datetime(year=int(ff['recording']['year'][()]), month=int(ff['recording']['month'][()]), day=int(ff['recording']['day'][()]), hour=int(ff['recording']['hour'][()]), minute=int(ff['recording']['minute'][()]), second=int(ff['recording']['second'][()]))
        all_ch_names = [''.join(map(chr, ff[ff['hdr']['signal_labels'][x,0]][()].flatten())) for x in range(len(ff['hdr']['signal_labels']))]
     
    # get channels to be read
    if ch_names is None:
        match_ch_ids = None
        subset_ch_names = all_ch_names
    else:
        match_ch_ids = [[xi for xi,x in enumerate(all_ch_names) if re.match(y.lower(),x.lower())] for y in ch_names]
        assert all([len(x)==1 for x in match_ch_ids])
        match_ch_ids = [x[0] for x in match_ch_ids]
        subset_ch_names = [all_ch_names[x] for x in match_ch_ids]
    with h5py.File(signal_path, 'r') as ff:
        signals = ff['s']
        T = signals.shape[0]
        if load_data:
            if match_ch_ids is None:
                signals = signals[()]
            else:
                signals = np.array([signals[:,x] for x in match_ch_ids]).T
    if load_data:
        signals = signals.T

    # get sleep stages
    stage2num = {'w':5, 'wake':5, 'rem':4, 'r':4, 'n1':3, 'n2':2, 'n3':1}
    pattern = re.compile('sleep_stage_(.*)', re.I)
    sleep_stages = np.zeros(T)+np.nan
    epoch_time = 30  # [second]
    epoch_size = int(round(epoch_time*Fs))
    epoch_start_ids = []
    for i in range(len(annot)):
        res = re.match(pattern, annot.event.iloc[i])
        if not res:
            continue
        stage = stage2num.get(res.group(1).lower(), np.nan)
        if pd.notna(stage):
            this_datetime = datetime.datetime.combine(start_time.date(), datetime.datetime.strptime(annot.time.iloc[i],'%H:%M:%S').time())
            if this_datetime<start_time:
                this_datetime += datetime.timedelta(days=1)
            start = int(round((this_datetime - start_time).total_seconds()*Fs))
            end   = start + epoch_size
            if 0<=start<end<=len(sleep_stages):
                sleep_stages[start:end] = stage
                epoch_start_ids.append(start)

    params = {'Fs':Fs, 'start_time':start_time, 'ch_names':subset_ch_names, 'epoch_start_ids':np.array(epoch_start_ids)}
    return signals, sleep_stages, params


def preprocess_EEG(signals, Fs):
    signals = signals - signals.mean(axis=1, keepdims=True)
    signals = mne.filter.notch_filter(signals, Fs, 60, verbose=False)
    signals = mne.filter.filter_data(signals, Fs, 0.3, 35, verbose=False)
    return signals


def removal_artifact_epochs(signals, epoch_start_ids, Fs):
    epoch_size = int(round(30*Fs))
    epochs = np.array([signals[:,x:x+epoch_size] for x in epoch_start_ids])
    good_ids = np.all(np.abs(epochs)<300, axis=(1,2))
    return epoch_start_ids[good_ids]


def get_features(signals, Fs, epoch_start_ids, th=0.89, return_internal=False):
    signals_f = mne.filter.filter_data(signals, Fs, 0.5, 2, verbose=False)
    
    # compute instantaneous variance explained by SWA    
    subepoch_size = int(round(2*Fs))+1
    move_var   = pd.DataFrame(signals.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T
    move_var_f = pd.DataFrame(signals_f.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T
    move_var_explained = move_var_f/move_var
    
    is_swa = np.zeros_like(signals, dtype=bool)
    is_swa2 = np.zeros_like(signals, dtype=bool)
    is_swa3 = np.zeros_like(signals, dtype=bool)
    for chi in range(len(signals_f)):
        # find zero-crossing
        ids_zc_down = np.where((signals_f[chi,:-1]>0)&(signals_f[chi,1:]<0))[0]
        ids_zc_up   = np.where((signals_f[chi,:-1]<0)&(signals_f[chi,1:]>0))[0]
        ids_zc = np.sort(np.unique(np.r_[0, ids_zc_down, ids_zc_up, signals.shape[-1]-1]))
        for i in range(len(ids_zc)-2):
            #sig_f = signals_f[chi, ids_zc[i]+1:ids_zc[i+2]+1]
            #sig = signals[chi, ids_zc[i]+1:ids_zc[i+2]+1]
            ve = move_var_explained[chi, ids_zc[i]+1:ids_zc[i+2]+1].max()
            if ve>th:#and sig_f.var()/sig.var()>0.5
                is_swa[chi, ids_zc[i]+1:ids_zc[i+2]+1] = True
        
        # fill short gaps and pad 2 seconds
        len_ = int(round(2*Fs))
        is_swa2[chi] = is_swa[chi]
        cc = 0
        for k,l in groupby(is_swa[chi]):
            ll = len(list(l))
            if not k and ll<=len_:
                is_swa2[chi, cc:cc+ll] = True
            elif k:
                is_swa2[chi, max(0,cc-len_):min(is_swa2.shape[1],cc+ll+len_)] = True
            cc += ll
        
        # only keep >75uV
        cc = 0
        for k,l in groupby(is_swa2[chi]):
            ll = len(list(l))
            if k and signals[chi,cc:cc+ll].max()-signals[chi,cc:cc+ll].min()>75:
                is_swa3[chi, cc:cc+ll] = True
            cc += ll
    
    epoch_size = int(round(30*Fs))
    is_swa3 = np.array([is_swa3[:,x:x+epoch_size] for x in epoch_start_ids])
    swa_perc = is_swa3.mean(axis=-1).min(axis=-1)
    
    """
    epoch_size = int(round(30*Fs))
    epochs_f = np.array([signals_f[:,x:x+epoch_size] for x in epoch_start_ids])
    epochs = np.array([signals[:,x:x+epoch_size] for x in epoch_start_ids])
    """
    #idx=np.where(sleep_stages[epoch_start_ids]==1)[0][10];sig2=np.array(epochs[idx,1]);sig2[~is_swa3[idx,1]]=np.nan;plt.close();fig=plt.figure(figsize=(10,4));ax=fig.add_subplot(111);ax.plot(tt,epochs[idx,1],c='k');ax.plot(tt,sig2,c='r');ax.set_ylim(-50,50);ax.set_yticks(np.arange(-50,60,10));ax.set_xlim(0,30);ax.set_xlabel('time (second)');ax.set_ylabel('F4-M1, microvolts');ax.yaxis.grid(True);seaborn.despine();plt.tight_layout();plt.savefig('example_N3_10.png')
    
    if return_internal:
        is_swa2 = np.array([is_swa2[:,x:x+epoch_size] for x in epoch_start_ids])
        move_var_explained = np.array([move_var_explained[:,x:x+epoch_size] for x in epoch_start_ids])
        return swa_perc, is_swa2, is_swa3, move_var_explained
    else:
        return swa_perc


if __name__=='__main__':
    outcome = 'Dementia'
    #covs = ['Age', 'Sex', 'Race', 'BMI', 'MedBenzo', 'MedAntiDep', 'MedSedative', 'MedAntiEplipetic', 'MedStimulant']

    df = pd.read_csv(f'../data/mastersheet_matched_{outcome}.csv')
    df['DOVshifted'] = pd.to_datetime(df.DOVshifted)
    df_annot = pd.read_csv('../data/annotations_sleep_stages.zip', compression='zip')
    df_annot['DOVshifted'] = pd.to_datetime(df_annot.DOVshifted)
    
    # get features
    """
    df_feat = defaultdict(list)
    for i in tqdm(range(len(df))):
        hashid = df.HashID.iloc[i]
        dov = df.DOVshifted.iloc[i]
        signals, sleep_stages, params = load_data(df.SignalPath.iloc[i].split('/')[-2], ch_names=['F3-M', 'F4-M'], df_annot=df_annot)
        Fs = params['Fs']
        epoch_start_ids = params['epoch_start_ids']

        signals = preprocess_EEG(signals, Fs)
        epoch_start_ids = removal_artifact_epochs(signals, epoch_start_ids, Fs)
        swa_perc = get_features(signals, Fs, epoch_start_ids)
        sleep_stages = sleep_stages[epoch_start_ids]

        df_feat['HashID'].extend([hashid]*len(sleep_stages))
        df_feat['DOVshifted'].extend([dov]*len(sleep_stages))
        df_feat['EpochStartIdx'].extend(epoch_start_ids)
        df_feat['SleepStage'].extend(sleep_stages)
        #for th in thresholds:
        #    df_feat[f'SWA_amp_{th}'].extend(swa_amp[th])
        #for th in thresholds:
        #    df_feat[f'SWA_perc_{th}'].extend(swa_perc[th])
        df_feat['SWA_perc'].extend(swa_perc)
        if i%10==0:
            pd.DataFrame(data=df_feat).to_csv('SWA_features.csv', index=False)
    df_feat = pd.DataFrame(data=df_feat)
    df_feat.to_csv('SWA_features.zip',compression='zip',index=False)
    """
    df_feat = pd.read_csv('SWA_features.zip', compression='zip')
    df_feat['DOVshifted'] = pd.to_datetime(df_feat.DOVshifted)
    
    #plt.close();fig=plt.figure(figsize=(4,4));ax=fig.add_subplot(111);ax.plot(fpr,tpr,c='k');ax.plot([0,1],[0,1],'r',ls='--');ax.set_xlim(-0.01,1.01);ax.set_ylim(-0.01,1.01);ax.text(0.99,0.01,f'AUC = {roc_auc_score(stages,swaperc):.2f}', ha='right', va='bottom');ax.scatter([fpr[idx]],[tpr[idx]],c='b',s=50);ax.text(fpr[idx]+0.03,tpr[idx],f'thres = {tt[idx]:.2f}',ha='left',va='top');ax.scatter([fpr[idx2]],[tpr[idx2]],c='b',s=50);ax.text(fpr[idx2]+0.03,tpr[idx2],f'thres = {tt[idx2]:.2f}',ha='left',va='top');ax.set_xlabel('FPR, 1-specificity');ax.set_ylabel('TPR, sensitivity');ax.grid(True);seaborn.despine();plt.tight_layout();plt.savefig('stage_n2n3_vs_swa_perc.png')
    
    #th=0.8;ids1=df_feat[f'SWA_perc_{th}'][df_feat.SleepStage==1].dropna().values;ids0=df_feat[f'SWA_perc_{th}'][df_feat.SleepStage==2].dropna().values;fpr,tpr,tt=roc_curve(np.r_[np.zeros_like(ids0),np.ones_like(ids1)],np.r_[ids0,ids1]);tt[np.argmin(fpr**2+(1-tpr)**2)]
    #plt.close();plt.plot(df_feat['SWA_amp_0.7'][df_feat.SleepStage==2].dropna());plt.ylim(0,200);plt.savefig('0.7_N2.png')
    
    sids = df[['HashID', 'DOVshifted']]
    X = []
    S = []
    for i in np.arange(len(df)):
        ids = (df_feat.HashID==df.HashID.iloc[i])&(df_feat.DOVshifted==df.DOVshifted.iloc[i])
        X.append(df_feat[['SWA_perc']][ids].values)
        S.append(df_feat['SleepStage'][ids].values)
    #Lnames = ['Age', 'Sex', 'Race', 'BMI']
    #L = df[Lnames].values  # check NaN
    Y = df[f'Y_{outcome}'].values
    import pdb;pdb.set_trace()
    
    # save
    with open(f'dataset_{outcome}.pickle', 'wb') as ff:
        pickle.dump({
            'sids':sids, 'X':X, 'S':S, 'Y':Y,# 'L':L,
            'Xnames':['SWA_perc'],# 'Lnames':Lnames,
            }, ff)
