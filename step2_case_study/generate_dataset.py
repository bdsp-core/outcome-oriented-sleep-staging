from itertools import product
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


def prepare_data(folder_path, ch_names=None, load_data=True):
    # read annotations file
    annot_folder = r'dropbox_mgh:/BDSP_engineering/new_redacted_annotations'
    annot_path = os.path.join(annot_folder, folder_path, folder_path+'_annotations.csv')
    output = subprocess.check_output(['rclone', 'cat', annot_path]).decode()
    annot = pd.read_csv(StringIO(output))

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
                signals = signals[:,match_ch_ids]  #TODO assume ascending order
    if load_data:
        signals = signals.T

    # get sleep stages
    stage2num = {'w':5, 'wake':5, 'rem':4, 'r':4, 'n1':3, 'n2':2, 'n3':1}
    pattern = re.compile('sleep_stage_(.*)', re.I)
    sleep_stages = np.zeros(T)+np.nan
    epoch_time = 30  # [second]
    epoch_size = int(round(epoch_time*Fs))
    sleep_epoch_start_ids = []
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
                sleep_epoch_start_ids.append(start)

    params = {'Fs':Fs, 'start_time':start_time, 'ch_names':subset_ch_names, 'sleep_epoch_start_ids':sleep_epoch_start_ids}
    return signals, sleep_stages, params


def preprocess_EEG(signals, Fs):
    signals = signals - signals.mean(axis=1, keepdims=True)
    signals = mne.filter.notch_filter(signals, Fs, 60, verbose=False)
    signals = mne.filter.filter_data(signals, Fs, 0.3, 35, verbose=False)
    return signals


def get_features(signals, Fs, epoch_start_ids, thresholds):
    signals_f = mne.filter.filter_data(signals, Fs, 0.5, 2, verbose=False)
    epoch_size = int(round(30*Fs))
    epochs_f = np.array([signals_f[:,x:x+epoch_size] for x in epoch_start_ids])
    epochs = np.array([signals[:,x:x+epoch_size] for x in epoch_start_ids])

    # compute instantaneous variance explained by SWA
    subepoch_size = int(round(2*Fs))+1
    move_var = pd.DataFrame(epochs.reshape(-1,epochs.shape[-1]).T).\
                rolling(subepoch_size, center=True, min_periods=1).var()\
                .values.T.reshape(*epochs.shape[:2],-1)
    move_var_f = pd.DataFrame(epochs_f.reshape(-1,epochs_f.shape[-1]).T).\
                rolling(subepoch_size, center=True, min_periods=1).var()\
                .values.T.reshape(*epochs_f.shape[:2],-1)
    move_var_explained = move_var_f/move_var

    swa_perc = {}
    swa_amp = {}
    min_swa_len = int(round(2*Fs))
    for th in thresholds:
        mask = move_var_explained>=th
        swa_perc[th] = mask.mean(axis=-1).max(axis=-1)
        swa_amp_ = np.zeros(epochs.shape[:2])+np.nan
        for i,j in product(range(mask.shape[0]), range(mask.shape[1])):
            sig = epochs[i,j][mask[i,j]]
            if len(sig)>min_swa_len:
                lb, ub = np.nanpercentile(sig, (2.5,97.5))
                swa_amp_[i,j] = ub-lb
        swa_amp[th] = np.nanmax(swa_amp_, axis=1)
    return swa_amp, swa_perc


if __name__=='__main__':
    outcome = 'Dementia'
    #covs = ['Age', 'Sex', 'Race', 'BMI', 'MedBenzo', 'MedAntiDep', 'MedSedative', 'MedAntiEplipetic', 'MedStimulant']

    df = pd.read_csv(f'../data/mastersheet_matched_{outcome}.csv')
    df['DOVshifted'] = pd.to_datetime(df.DOVshifted)

    """
    # get features
    thresholds = [0.8,0.81,0.82,0.83]
    df_feat = defaultdict(list)
    for i in tqdm(range(len(df))):
        hashid = df.HashID.iloc[i]
        dov = df.DOVshifted.iloc[i]
        signals, sleep_stages, params = prepare_data(df.SignalPath.iloc[i].split('/')[-2], ch_names=['F3-M', 'F4-M'])
        Fs = params['Fs']
        sleep_epoch_start_ids = params['sleep_epoch_start_ids']
        sleep_stages = sleep_stages[sleep_epoch_start_ids]

        signals = preprocess_EEG(signals, Fs)
        swa_amp, swa_perc = get_features(signals, Fs, sleep_epoch_start_ids, thresholds)

        df_feat['HashID'].extend([hashid]*len(sleep_stages))
        df_feat['DOVshifted'].extend([dov]*len(sleep_stages))
        df_feat['Epoch'].extend(np.arange(len(sleep_stages))+1)
        df_feat['SleepStage'].extend(sleep_stages)
        for th in thresholds:
            df_feat[f'SWA_amp_{th}'].extend(swa_amp[th])
        for th in thresholds:
            df_feat[f'SWA_perc_{th}'].extend(swa_perc[th])
        if i%10==0:
            pd.DataFrame(data=df_feat).to_csv('SWA_features_different_thresholds.csv', index=False)
    df_feat = pd.DataFrame(data=df_feat)
    df_feat.to_csv('SWA_features_different_thresholds.zip',compression='zip',index=False)
    """
    df_feat = pd.read_csv('SWA_features_different_thresholds.zip', compression='zip')
    df_feat['DOVshifted'] = pd.to_datetime(df_feat.DOVshifted)
    
    #th=0.8;ids1=df_feat[f'SWA_perc_{th}'][df_feat.SleepStage==1].dropna().values;ids0=df_feat[f'SWA_perc_{th}'][df_feat.SleepStage==2].dropna().values;fpr,tpr,tt=roc_curve(np.r_[np.zeros_like(ids0),np.ones_like(ids1)],np.r_[ids0,ids1]);tt[np.argmin(fpr**2+(1-tpr)**2)]
    #plt.close();plt.plot(df_feat['SWA_amp_0.7'][df_feat.SleepStage==2].dropna());plt.ylim(0,200);plt.savefig('0.7_N2.png')
    
    thres = 0.82  # by making sure the best separating point is SWA perc=0.2
    
    sids = df[['HashID', 'DOVshifted']]
    X = []
    S = []
    for i in np.arange(len(df)):
        ids = (df_feat.HashID==df.HashID.iloc[i])&(df_feat.DOVshifted==df.DOVshifted.iloc[i])
        X.append(df_feat[[f'SWA_amp_{thres}', f'SWA_perc_{thres}']][ids].fillna(0).values)
        S.append(df_feat['SleepStage'][ids].values)
    #Lnames = ['Age', 'Sex', 'Race', 'BMI']
    #L = df[Lnames].values  # check NaN
    Y = df[f'Y_{outcome}'].values
    
    # save
    with open(f'dataset_{outcome}.pickle', 'wb') as ff:
        pickle.dump({
            'sids':sids, 'X':X, 'S':S, 'Y':Y,# 'L':L,
            'Xnames':['SWA_amp', 'SWA_perc'],# 'Lnames':Lnames,
            }, ff)
