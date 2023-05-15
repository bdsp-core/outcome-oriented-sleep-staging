from itertools import groupby
from collections import defaultdict
import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.signal.windows import parzen
from tqdm import tqdm
import mne
import sys
sys.path.insert(0, '..')
from myfunctions import load_data, filter_signal
from check_detection import load_data, filter_signal


def remove_bad_start_end(signals, sleep_stages, Fs, epoch_time):
    epoch_size = int(round(epoch_time*Fs))
    epoch_start_ids = np.arange(0, signals.shape[1]-epoch_size+1, epoch_size)
    sleep_stages_epoch = sleep_stages[epoch_start_ids]
    epochs = np.array([signals[:,x:x+epoch_size] for x in epoch_start_ids])
    good_ids = np.all(np.abs(epochs)<500, axis=(1,2))&pd.notna(sleep_stages[epoch_start_ids])
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
    sleep_stages_epoch = sleep_stages_epoch[good_ids2]
    signals = signals[:, epoch_start_ids.min():epoch_start_ids.max()+epoch_size]
    sleep_stages = sleep_stages[epoch_start_ids.min():epoch_start_ids.max()+epoch_size]
    return signals, sleep_stages, sleep_stages_epoch, epoch_start_ids


def get_features(signals, sleep_stages, Fs, epoch_start_ids, epoch_time, n_jobs=1):
    epoch_size = int(round(Fs*epoch_time))
    eeg_ch_names_f = ['F3-M2', 'F4-M1']
    eeg_ch_names_c = ['C3-M2', 'C4-M1']
    eeg_ch_names_o = ['O1-M2', 'O2-M1']
    eeg_ch_names = eeg_ch_names_f + eeg_ch_names_c + eeg_ch_names_o
    eeg = signals[eeg_ch_names].values.T
    tt = np.arange(len(epoch_start_ids))*epoch_time
    tt_sample = np.round(tt*Fs).astype(int)

    ## relative band power

    epochs = np.array([eeg[:,x:x+epoch_size] for x in tt_sample])
    spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs, fmin=0.3, fmax=35,
        bandwidth=0.5, adaptive=False, low_bias=True, normalization='full', n_jobs=n_jobs, verbose=False)
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

    spindle_peak_freq = get_spindle_peak_freq(spec, freq, sleep_stages[tt_sample])
    spindle_peak_freq[np.isnan(spindle_peak_freq)] = np.nanmedian(spindle_peak_freq)
    spindle_res = my_spindle_detect(
        eeg, sleep_stages, Fs, eeg_ch_names,
        include=[2,1],
        freq_sp=[[x-1,x+1] for x in spindle_peak_freq],
        thresh={'corr':0.74, 'rel_pow':0.07, 'rms':1},
        verbose=False)

    has_spindle_F = np.zeros_like(epoch_start_ids)
    has_spindle_C = np.zeros_like(epoch_start_ids)
    has_spindle_O = np.zeros_like(epoch_start_ids)
    for i in range(len(spindle_res)):
        idx = np.searchsorted(tt, spindle_res.Peak.iloc[i])-1
        chn = spindle_res.Channel.iloc[i]
        if chn in eeg_ch_names_f:
            has_spindle_F[idx] = 1
        elif chn in eeg_ch_names_c:
            has_spindle_C[idx] = 1
        elif chn in eeg_ch_names_o:
            has_spindle_O[idx] = 1
        
    ## SWA amp and perc

    sw_res = my_sw_detect(
            eeg, sleep_stages, Fs, eeg_ch_names,
            include=[2,1], amp_ptp=[70, 350],
            verbose=False)

    sw_amp_ptp_F = np.zeros_like(epoch_start_ids)
    sw_amp_ptp_C = np.zeros_like(epoch_start_ids)
    sw_amp_ptp_O = np.zeros_like(epoch_start_ids)
    sw_mask = np.zeros((len(eeg_ch_names), len(epoch_start_ids)), dtype=bool)
    for i in range(len(sw_res)):
        idx = np.searchsorted(tt, sw_res.MidCrossing.iloc[i])-1
        chn = sw_res.Channel.iloc[i]
        if chn in eeg_ch_names_f:
            sw_amp_ptp_F[idx] = max(sw_amp_ptp_F[idx], sw_res.PTP.iloc[i])
        elif chn in eeg_ch_names_c:
            sw_amp_ptp_C[idx] = max(sw_amp_ptp_C[idx], sw_res.PTP.iloc[i])
        elif chn in eeg_ch_names_o:
            sw_amp_ptp_O[idx] = max(sw_amp_ptp_O[idx], sw_res.PTP.iloc[i])
        chid = sw_res.IdxChannel.iloc[i]
        start = int(round(sw_res.Start.iloc[i]*Fs))
        end   = int(round(sw_res.End.iloc[i]*Fs))
        sw_mask[chid, start:end] = True
    sw_perc_F = np.zeros_like(epoch_start_ids)
    sw_perc_C = np.zeros_like(epoch_start_ids)
    sw_perc_O = np.zeros_like(epoch_start_ids)
    for i, start in enumerate(epoch_start_ids):
        sw_perc_F[i] = sw_mask[[0,1], start:start+epoch_size].mean()  #TODO: [0,1]
        sw_perc_C[i] = sw_mask[[2,3], start:start+epoch_size].mean()  #TODO: [0,1]
        sw_perc_O[i] = sw_mask[[4,5], start:start+epoch_size].mean()  #TODO: [0,1]

    ## rapid eye movement

    eog_ch_names = ['EOGL', 'EOGR']
    eog = signals[eog_ch_names].values.T
    rem_res = my_rem_detect(eog[0], eog[1], sleep_stages, Fs, eog_ch_name, include=[4], var_thres=0.7, verbose=False)

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
        'sw_perc_F':sw_perc_F, 'sw_perc_C':sw_perc_C, 'sw_perc_O':sw_perc_O,
        'has_rem':has_rem, 'emg_env_rank_mean':envelope2,
        })
    
    return df_res


if __name__=='__main__':
    outcome = 'Dementia'
    epoch_time = 30
    n_jobs = 14

    df = pd.read_csv(f'../data/mastersheet_matched_{outcome}.csv')
    df['DOVshifted'] = pd.to_datetime(df.DOVshifted)
    
    save_path = f'features_epoch{epoch_time}s.csv'
    if not os.path.exists(save_path):
        base_folder = '/bdsp/opendata/PSG/data/S0001'

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

            signals, sleep_stages, params = load_data(df.SignalPath.iloc[i].split('/')[-2], base_folder, ch_names_re, df_annot=df_annot)
            Fs = params['Fs']
            signals = signals.values.T

            signals = np.concatenate([
                filter_signal(signals[:6], Fs, 60, [0.3, 35]),
                filter_signal(signals[6:8], Fs, 60, [0.3, 35]),
                filter_signal(signals[8:], Fs, 60, [10, None]),
            ], axis=0)
            
            signals, sleep_stages, sleep_stages_epoch, epoch_start_ids = remove_bad_start_end(signals, sleep_stages, Fs, epoch_time)
            #TODO artifact removal epoch_start_ids = removal_artifact_epochs(signals, sleep_stages, epoch_start_ids, Fs)

            features = get_features(pd.DataFrame(data=signals.T, columns=ch_names), sleep_stages, Fs, epoch_start_ids, epoch_time, n_jobs=n_jobs)

            df_feat['HashID'].extend([hashid]*len(features))
            df_feat['DOVshifted'].extend([dov]*len(features))
            df_feat['EpochStartIdx'].extend(epoch_start_ids)
            df_feat['SleepStage'].extend(sleep_stages_epoch)
            for col in features.columns:
                df_feat[col].extend(features[col])
            if i%10==0:
                pd.DataFrame(data=df_feat).to_csv(save_path, index=False)
        df_feat = pd.DataFrame(data=df_feat)
        df_feat.to_csv(save_path, index=False)
    else:
        df_feat = pd.read_csv(save_path)
        df_feat['DOVshifted'] = pd.to_datetime(df_feat.DOVshifted)
    
    #plt.close();fig=plt.figure(figsize=(4,4));ax=fig.add_subplot(111);ax.plot(fpr,tpr,c='k');ax.plot([0,1],[0,1],'r',ls='--');ax.set_xlim(-0.01,1.01);ax.set_ylim(-0.01,1.01);ax.text(0.99,0.01,f'AUC = {roc_auc_score(stages,swaperc):.2f}', ha='right', va='bottom');ax.scatter([fpr[idx]],[tpr[idx]],c='b',s=50);ax.text(fpr[idx]+0.03,tpr[idx],f'thres = {tt[idx]:.2f}',ha='left',va='top');ax.scatter([fpr[idx2]],[tpr[idx2]],c='b',s=50);ax.text(fpr[idx2]+0.03,tpr[idx2],f'thres = {tt[idx2]:.2f}',ha='left',va='top');ax.set_xlabel('FPR, 1-specificity');ax.set_ylabel('TPR, sensitivity');ax.grid(True);seaborn.despine();plt.tight_layout();plt.savefig('stage_n2n3_vs_swa_perc.png')
    
    #th=0.8;ids1=df_feat[f'SWA_perc_{th}'][df_feat.SleepStage==1].dropna().values;ids0=df_feat[f'SWA_perc_{th}'][df_feat.SleepStage==2].dropna().values;fpr,tpr,tt=roc_curve(np.r_[np.zeros_like(ids0),np.ones_like(ids1)],np.r_[ids0,ids1]);tt[np.argmin(fpr**2+(1-tpr)**2)]
    #plt.close();plt.plot(df_feat['SWA_amp_0.7'][df_feat.SleepStage==2].dropna());plt.ylim(0,200);plt.savefig('0.7_N2.png')
    
    Xnames = list(df_feat.columns)
    Xnames.remove('HashID')
    Xnames.remove('DOVshifted')
    Xnames.remove('EpochStartIdx')
    Xnames.remove('SleepStage')
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
    with open(f'dataset_{outcome}_epoch{epoch_time}s2.pickle', 'wb') as ff:
        pickle.dump({
            'sids':sids, 'X':X, 'S':S, 'Y':Y, 'L':L,
            'Xnames':Xnames, 'Lnames':Lnames,
            }, ff)
