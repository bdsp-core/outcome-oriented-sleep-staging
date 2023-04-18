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
from scipy.signal import hilbert
from scipy.signal.windows import parzen
from tqdm import tqdm
import mne
import h5py
import yasa


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
        this_datetime = datetime.datetime.combine(start_time.date(), datetime.datetime.strptime(annot.time.iloc[i],'%H:%M:%S').time())
        if this_datetime<start_time:
            this_datetime += datetime.timedelta(days=1)
        start = int(round((this_datetime - start_time).total_seconds()*Fs))
        end   = start + epoch_size
        if 0<=start<end<=len(sleep_stages):
            if pd.notna(stage):
                sleep_stages[start:end] = stage
            epoch_start_ids.append(start)

    params = {'Fs':Fs, 'start_time':start_time, 'ch_names':subset_ch_names, 'epoch_start_ids':np.array(epoch_start_ids)}
    return signals, sleep_stages, params


def preprocess_signals(signals, Fs, notch_freq, bandpass_freq):
    signals = signals - signals.mean(axis=1, keepdims=True)
    signals = mne.filter.notch_filter(signals, Fs, notch_freq, verbose=False)
    signals = mne.filter.filter_data(signals, Fs, bandpass_freq[0], bandpass_freq[1], verbose=False)
    return signals


def remove_bad_start_end(signals, sleep_stages, epoch_start_ids, Fs, epoch_time):
    epoch_size = int(round(epoch_time*Fs))
    epochs = np.array([signals[:,x:x+epoch_size] for x in epoch_start_ids])
    good_ids = np.all(np.abs(epochs)<500, axis=(1,2))&pd.notna(sleep_stages)
    # only remove the start and end bad ones
    good_ids2 = np.ones_like(good_ids)
    cc = 0
    for k,l in groupby(good_ids):
        ll = len(list(l))
        if k==0 and cc==0:
            good_ids2[:ll] = False
        break
    cc = 0
    for k,l in groupby(good_ids):
        ll = len(list(l))
        if k==0 and cc+ll==len(good_ids):
            good_ids2[-ll:] = False
            break
        cc += ll

    epoch_start_ids = epoch_start_ids[good_ids2]
    signals = signals[:, epoch_start_ids.min():epoch_start_ids.max()+epoch_size]
    sleep_stages = sleep_stages[good_ids2]
    return signals, sleep_stages, epoch_start_ids


def sw_detect(eeg, Fs, ch_names, hypno=None, include=None, freq=[0.5, 2], freq_broad=None, thresh=0.89):
    # compute instantaneous variance explained by SWA
    eeg_swa = mne.filter.filter_data(eeg, Fs, freq[0], freq[1], verbose=False)
    subepoch_size = int(round(2*Fs))+1
    move_var   = pd.DataFrame(eeg.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T
    move_var_f = pd.DataFrame(eeg_swa.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T
    move_var_explained = move_var_f/move_var
    
    is_swa = np.zeros_like(eeg, dtype=bool)
    is_swa2 = np.zeros_like(eeg, dtype=bool)
    is_swa3 = np.zeros_like(eeg, dtype=bool)
    df_res = defaultdict(list)
    for chi in range(len(eeg_swa)):
        # find zero-crossing
        ids_zc_down = np.where((eeg_swa[chi,:-1]>0)&(eeg_swa[chi,1:]<0))[0]
        ids_zc_up   = np.where((eeg_swa[chi,:-1]<0)&(eeg_swa[chi,1:]>0))[0]
        ids_zc = np.sort(np.unique(np.r_[0, ids_zc_down, ids_zc_up, eeg.shape[-1]-1]))
        for i in range(len(ids_zc)-2):
            #sig_f = eeg_swa[chi, ids_zc[i]+1:ids_zc[i+2]+1]
            #sig = eeg[chi, ids_zc[i]+1:ids_zc[i+2]+1]
            ve = move_var_explained[chi, ids_zc[i]+1:ids_zc[i+2]+1].max()
            if 1>ve>swa_thres:#and sig_f.var()/sig.var()>0.5
                is_swa[chi, ids_zc[i]+1:ids_zc[i+2]+1] = True
        # fill short gaps <= 2 seconds
        len_ = int(round(2*Fs))
        is_swa2[chi] = is_swa[chi]
        cc = 0
        for k,l in groupby(is_swa[chi]):
            ll = len(list(l))
            if not k and ll<=len_:
                is_swa2[chi, cc:cc+ll] = True
            cc += ll
        
        cc = 0
        for k,l in groupby(is_swa2[chi]):
            ll = len(list(l))
            if k:
                df_res['Start'].append(cc/Fs)
                df_res['End'].append((cc+ll)/Fs)
                df_res['AmpPTP'].append(eeg[chi,cc:cc+ll].max()-eeg[chi,cc:cc+ll].min())
            cc += ll

    df_res = pd.DataFrame(df_res)
    return df_res


def get_features(signals, sleep_stages, Fs, epoch_start_ids, epoch_time):
    epoch_size = int(round(Fs*epoch_time))
    eeg_ch_names = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
    eeg = signals[eeg_ch_names].values.T
    tt = np.arange(len(epoch_start_ids))*epoch_time
    tt_sample = np.round(tt*Fs).astype(int)

    ## relative band power

    epochs = np.array([eeg[:,x:x+epoch_size] for x in tt_sample])
    spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs, fmin=0.3, fmax=35,
        bandwidth=0.5, adaptive=False, low_bias=True, normalization='full', n_jobs=8, verbose=False)
    theta_ids = (freq>=4)&(freq<8)
    alpha_ids = (freq>=8)&(freq<12)
    theta_rel_bp = spec[...,theta_ids].sum(axis=-1)/spec.sum(axis=-1)
    alpha_rel_bp = spec[...,alpha_ids].sum(axis=-1)/spec.sum(axis=-1)
    theta_rel_bp_F = theta_rel_bp[:,[0,1]].mean(axis=1)
    theta_rel_bp_C = theta_rel_bp[:,[2,3]].mean(axis=1)
    theta_rel_bp_O = theta_rel_bp[:,[4,5]].mean(axis=1)
    alpha_rel_bp_F = alpha_rel_bp[:,[0,1]].mean(axis=1)
    alpha_rel_bp_C = alpha_rel_bp[:,[2,3]].mean(axis=1)
    alpha_rel_bp_O = alpha_rel_bp[:,[4,5]].mean(axis=1)

    ## spindle
    """
-2 = Unscored
-1 = Artefact / Movement
0 = Wake
1 = N1 sleep
2 = N2 sleep
3 = N3 sleep
4 = REM sleep
    """
    stage_mapping = {5:0,4:4,3:1,2:2,1:3,-1:-1}
    sleep_stages[np.isnan(sleep_stages)] = -1
    applyall = np.vectorize(lambda x:stage_mapping.get(x,-1))
    sleep_stages2 = applyall(sleep_stages)
    sleep_stages2 = np.repeat(sleep_stages2, epoch_size)

    spindle_res = yasa.spindles_detect( eeg, sf=Fs, ch_names=eeg_ch_names,
        hypno=sleep_stages2, include=[2,3], freq_sp=[11,16], freq_broad=[1,30],
        duration=[0.5,2], min_distance=500,
        thresh={'corr': 0.65, 'rel_pow': 0.2, 'rms': 1.5},
        multi_only=False, remove_outliers=False, verbose=False)
    spindle_res = spindle_res.summary()

    has_spindle_F = np.zeros_like(epoch_start_ids)
    has_spindle_C = np.zeros_like(epoch_start_ids)
    has_spindle_O = np.zeros_like(epoch_start_ids)
    for i in range(len(spindle_res)):
        idx = np.searchsorted(tt, spindle_res.Peak.iloc[i])-1
        if spindle_res.Channel.iloc[i] in ['F3-M2', 'F4-M1']:
            has_spindle_F[idx] = 1
        elif spindle_res.Channel.iloc[i] in ['C3-M2', 'C4-M1']:
            has_spindle_C[idx] = 1
        elif spindle_res.Channel.iloc[i] in ['O1-M2', 'O2-M1']:
            has_spindle_O[idx] = 1
        
    ## SWA amp and perc

    #sw_res = sw_detect( eeg, Fs, eeg_ch_names,
    #    hypno=sleep_stages2, include=[2,3],
    #    freq=[0.5, 2], freq_broad=None, thresh=0.89) 
    sw_res = yasa.sw_detect(eeg, sf=Fs, ch_names=eeg_ch_names,
        hypno=sleep_stages2, include=[2,3], freq_sw=[0.5,2],
        dur_neg=[0.3,1.5], dur_pos=[0.1,1],
        amp_neg=[40,200], amp_pos=[10,150], amp_ptp=[50,350],
        remove_outliers=False, verbose=False)
    sw_res = sw_res.summary()
    sw_amp_ptp_F = np.zeros_like(epoch_start_ids)
    sw_amp_ptp_C = np.zeros_like(epoch_start_ids)
    sw_amp_ptp_O = np.zeros_like(epoch_start_ids)
    #TODO sw_perc_F50 = np.zeros_like(epoch_start_ids)
    for i in range(len(sw_res)):
        idx = np.searchsorted(tt, sw_res.MidCrossing.iloc[i])-1
        if sw_res.Channel.iloc[i] in ['F3-M2', 'F4-M1']:
            sw_amp_ptp_F[idx] = max(sw_amp_ptp_F[idx], sw_res.PTP.iloc[i])
        elif sw_res.Channel.iloc[i] in ['C3-M2', 'C4-M1']:
            sw_amp_ptp_C[idx] = max(sw_amp_ptp_C[idx], sw_res.PTP.iloc[i])
        elif sw_res.Channel.iloc[i] in ['O1-M2', 'O2-M1']:
            sw_amp_ptp_O[idx] = max(sw_amp_ptp_O[idx], sw_res.PTP.iloc[i])

    ## rapid eye movement

    eog_ch_names = ['EOGL', 'EOGR']
    eog = signals[eog_ch_names].values.T
    rem_res = yasa.rem_detect(eog[0], eog[1], Fs,
        hypno=sleep_stages2, include=[1,2,3,4],
        amplitude=[50,325], duration=[0.3,1.2],
        freq_rem=[0.5,5], remove_outliers=False, verbose=False)
    rem_res = rem_res.summary()
    has_rem = np.zeros_like(epoch_start_ids)
    for i in range(len(rem_res)):
        idx = np.searchsorted(tt, rem_res.Peak.iloc[i])-1
        has_rem[idx] = 1
    
    ## EMG

    emg_ch_names = 'EMG'
    emg = signals[emg_ch_names].values
    envelope = np.abs(hilbert(emg))
    envelope = np.convolve(envelope, parzen(200*4)/np.sum(parzen(200*4))*2.5,mode='same')
    levels = np.percentile(envelope, np.arange(0,101,10))
    levels[0] = -np.inf
    envelope2 = np.searchsorted(levels, envelope)-1
    envelope2 = np.array([envelope2[x:x+epoch_size] for x in tt_sample])
    envelope2 = envelope2.mean(axis=1)

    df_res = pd.DataFrame(data={
        'alpha_rel_bp_F':alpha_rel_bp_F, 'alpha_rel_bp_C':alpha_rel_bp_C, 'alpha_rel_bp_O':alpha_rel_bp_O,
        'theta_rel_bp_F':theta_rel_bp_F, 'theta_rel_bp_C':theta_rel_bp_C, 'theta_rel_bp_O':theta_rel_bp_O,
        'has_spindle_F':has_spindle_F, 'has_spindle_C':has_spindle_C, 'has_spindle_O':has_spindle_O,
        'sw_amp_ptp_F':sw_amp_ptp_F, 'sw_amp_ptp_C':sw_amp_ptp_C, 'sw_amp_ptp_O':sw_amp_ptp_O,
        'has_rem':has_rem, 'emg_env_rank_mean':envelope2,
        })
    
    return df_res


if __name__=='__main__':
    outcome = 'Dementia'
    epoch_time = 30

    df = pd.read_csv(f'../data/mastersheet_matched_{outcome}.csv')
    df['DOVshifted'] = pd.to_datetime(df.DOVshifted)
    df_annot = pd.read_csv('../data/annotations_sleep_stages.zip', compression='zip')
    df_annot['DOVshifted'] = pd.to_datetime(df_annot.DOVshifted)

    eeg_ch_names = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
    eog_ch_names = ['EOGL', 'EOGR']
    emg_ch_names = ['EMG']
    ch_names = eeg_ch_names + eog_ch_names + emg_ch_names
    ch_names_re = ['F3-', 'F4-', 'C3-', 'C4-', 'O1-', 'O2-', 'E1-', 'E2-', 'chin']
    
    # get features
    df_feat = defaultdict(list)
    for i in tqdm(range(len(df))):
        hashid = df.HashID.iloc[i]
        dov = df.DOVshifted.iloc[i]

        signals, sleep_stages, params = load_data(df.SignalPath.iloc[i].split('/')[-2], ch_names=ch_names_re, df_annot=df_annot)
        Fs = params['Fs']
        epoch_start_ids = params['epoch_start_ids']
        sleep_stages = sleep_stages[epoch_start_ids]

        signals = np.concatenate([
            preprocess_signals(signals[:6], Fs, 60, [0.3, 35]),
            preprocess_signals(signals[6:8], Fs, 60, [0.3, 35]),
            preprocess_signals(signals[8:], Fs, 60, [10, None]),
        ], axis=0)
        signals, sleep_stages, epoch_start_ids = remove_bad_start_end(signals, sleep_stages, epoch_start_ids, Fs, epoch_time)
        #TODO artifact removal epoch_start_ids = removal_artifact_epochs(signals, sleep_stages, epoch_start_ids, Fs)

        features = get_features(pd.DataFrame(data=signals.T, columns=ch_names), sleep_stages, Fs, epoch_start_ids, epoch_time)

        df_feat['HashID'].extend([hashid]*len(sleep_stages))
        df_feat['DOVshifted'].extend([dov]*len(sleep_stages))
        df_feat['EpochStartIdx'].extend(epoch_start_ids)
        df_feat['SleepStage'].extend(sleep_stages)
        for col in features.columns:
            df_feat[col].append(features[col].iloc[0])
        if i%10==0:
            pd.DataFrame(data=df_feat).to_csv(f'features_{outcome}_epoch{epoch_time}s.csv', index=False)
    df_feat = pd.DataFrame(data=df_feat)
    df_feat.to_csv(f'features_{outcome}_epoch{epoch_time}s.csv',index=False)
    """
    df_feat = pd.read_csv(f'features_{outcome}_epoch{epoch_time}s.csv')
    df_feat['DOVshifted'] = pd.to_datetime(df_feat.DOVshifted)
    """
    
    #plt.close();fig=plt.figure(figsize=(4,4));ax=fig.add_subplot(111);ax.plot(fpr,tpr,c='k');ax.plot([0,1],[0,1],'r',ls='--');ax.set_xlim(-0.01,1.01);ax.set_ylim(-0.01,1.01);ax.text(0.99,0.01,f'AUC = {roc_auc_score(stages,swaperc):.2f}', ha='right', va='bottom');ax.scatter([fpr[idx]],[tpr[idx]],c='b',s=50);ax.text(fpr[idx]+0.03,tpr[idx],f'thres = {tt[idx]:.2f}',ha='left',va='top');ax.scatter([fpr[idx2]],[tpr[idx2]],c='b',s=50);ax.text(fpr[idx2]+0.03,tpr[idx2],f'thres = {tt[idx2]:.2f}',ha='left',va='top');ax.set_xlabel('FPR, 1-specificity');ax.set_ylabel('TPR, sensitivity');ax.grid(True);seaborn.despine();plt.tight_layout();plt.savefig('stage_n2n3_vs_swa_perc.png')
    
    #th=0.8;ids1=df_feat[f'SWA_perc_{th}'][df_feat.SleepStage==1].dropna().values;ids0=df_feat[f'SWA_perc_{th}'][df_feat.SleepStage==2].dropna().values;fpr,tpr,tt=roc_curve(np.r_[np.zeros_like(ids0),np.ones_like(ids1)],np.r_[ids0,ids1]);tt[np.argmin(fpr**2+(1-tpr)**2)]
    #plt.close();plt.plot(df_feat['SWA_amp_0.7'][df_feat.SleepStage==2].dropna());plt.ylim(0,200);plt.savefig('0.7_N2.png')
    
    Xnames = []
    Lnames = ['Age', 'Sex', 'Race', 'BMI', 'MedBenzo', 'MedAntiDep', 'MedSedative', 'MedAntiEplipetic', 'MedStimulant']
    sids = df[['HashID', 'DOVshifted']]
    X = []
    S = []
    for i in np.arange(len(df)):
        ids = (df_feat.HashID==df.HashID.iloc[i])&(df_feat.DOVshifted==df.DOVshifted.iloc[i])
        X.append(df_feat[Xnames][ids].values)
        S.append(df_feat['SleepStage'][ids].values)
    L = df[Lnames].values  # check NaN
    Y = df[f'Y_{outcome}'].values
    
    # save
    with open(f'dataset_{outcome}_epoch{epoch_time}s.pickle', 'wb') as ff:
        pickle.dump({
            'sids':sids, 'X':X, 'S':S, 'Y':Y, 'L':L,
            'Xnames':Xnames, 'Lnames':Lnames,
            }, ff)
