from itertools import groupby
from collections import defaultdict
import os
import pickle
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert
from scipy.signal.windows import parzen
from tqdm import tqdm
import mne
import sys
sys.path.insert(0, '..')
from myfunctions import load_data, filter_signal
from pattern_detection import get_spindle_peak_freq, my_spindle_detect, my_sw_detect, my_rem_detect


def remove_bad_start_end(signals, sleep_stages):
    # only remove the start and end bad ones
    good_ids = pd.notna(sleep_stages)
    good_ids2 = np.ones_like(good_ids)
    cc = 0
    for k,l in groupby(good_ids):
        ll = len(list(l))
        if not k and cc==0:
            good_ids2[:ll] = False
        break
    cc = 0
    for k,l in groupby(good_ids):
        ll = len(list(l))
        if not k and cc+ll==len(good_ids):
            good_ids2[-ll:] = False
            break
        cc += ll

    signals = signals[:, good_ids2]
    sleep_stages = sleep_stages[good_ids2]
    return signals, sleep_stages


def get_features(signals, sleep_stages, Fs, epoch_times, n_jobs=1):
    only_one_epoch_time = type(epoch_times) in [int, float]
    if only_one_epoch_time:
        epoch_times = [epoch_times]

    eeg_ch_names_f = ['F3-M2', 'F4-M1']
    eeg_ch_names_c = ['C3-M2', 'C4-M1']
    eeg_ch_names_o = ['O1-M2', 'O2-M1']
    eeg_ch_names = eeg_ch_names_f + eeg_ch_names_c + eeg_ch_names_o
    eeg = signals[eeg_ch_names].values.T

    subepoch_size = int(round(2*Fs))+1
    move_var = pd.DataFrame(eeg.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T

    eeg_theta = mne.filter.filter_data(eeg, Fs, 4, 8, verbose=False)
    move_var_theta = pd.DataFrame(eeg_theta.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T
    move_theta_rel_bp = move_var_theta/move_var
    move_theta_rel_bp[move_theta_rel_bp>=1] = np.nan

    eeg_alpha = mne.filter.filter_data(eeg, Fs, 8, 12, verbose=False)
    move_var_alpha = pd.DataFrame(eeg_alpha.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T
    move_alpha_rel_bp = move_var_alpha/move_var
    move_alpha_rel_bp[move_alpha_rel_bp>=1] = np.nan

    sw_res = my_sw_detect(
            eeg, sleep_stages, Fs, eeg_ch_names,
            include=[2,1], amp_ptp=[70, 350],
            verbose=False)
    sw_mask = np.zeros_like(eeg, dtype=bool)
    for i in range(len(sw_res)):
        chid = sw_res.IdxChannel.iloc[i]
        start = int(round(sw_res.Start.iloc[i]*Fs))
        end   = int(round(sw_res.End.iloc[i]*Fs))
        sw_mask[chid, start:end] = True

    eog_ch_names = ['EOGL', 'EOGR']#TODO
    eog = signals[eog_ch_names].values.T
    rem_res = my_rem_detect(eog[0], eog[1], sleep_stages, Fs, eog_ch_names[0], include=[4], var_thres=0.7, verbose=False)

    emg_ch_names = 'EMG'#TODO
    emg = signals[emg_ch_names].values
    L = len(sleep_stages)
    nfast = next_fast_len(L)
    env = np.abs(hilbert(emg, N=nfast)[:L])
    window = parzen(int(round(Fs*4)))
    env = np.convolve(env, window/np.sum(window)*2.5,mode='same')
    levels = np.percentile(env, np.arange(0,101,10))
    levels[0] = -np.inf
    env = np.searchsorted(levels, env)-1

    df_feats = {}
    sleep_stages_epochs = {}
    epoch_start_idss = {}
    for ei, epoch_time in enumerate(epoch_times):
        epoch_size = int(round(epoch_time*Fs))
        epoch_start_ids = np.arange(0, eeg.shape[1]-epoch_size+1, epoch_size)
        tt = np.arange(len(epoch_start_ids))*epoch_time
        tt_sample = np.round(tt*Fs).astype(int)
        sleep_stages_epoch = sleep_stages[epoch_start_ids]

        ## relative band power

        epochs_theta_rel_bp = np.array([move_theta_rel_bp[:,x:x+epoch_size] for x in tt_sample])
        theta_rel_bp_F = np.nanmean(epochs_theta_rel_bp[:,[0,1]], axis=(1,2))
        theta_rel_bp_C = np.nanmean(epochs_theta_rel_bp[:,[2,3]], axis=(1,2))
        theta_rel_bp_O = np.nanmean(epochs_theta_rel_bp[:,[4,5]], axis=(1,2))
        epochs_alpha_rel_bp = np.array([move_alpha_rel_bp[:,x:x+epoch_size] for x in tt_sample])
        alpha_rel_bp_F = np.nanmean(epochs_alpha_rel_bp[:,[0,1]], axis=(1,2))
        alpha_rel_bp_C = np.nanmean(epochs_alpha_rel_bp[:,[2,3]], axis=(1,2))
        alpha_rel_bp_O = np.nanmean(epochs_alpha_rel_bp[:,[4,5]], axis=(1,2))

        ## spindle

        if ei==0:
            assert epoch_time==30
            epochs = np.array([eeg[:,x:x+epoch_size] for x in tt_sample])
            spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs, fmin=0.3, fmax=35,
                bandwidth=0.5, adaptive=False, low_bias=True, normalization='full', n_jobs=n_jobs, verbose=False)
            spindle_peak_freq = get_spindle_peak_freq(spec, freq, sleep_stages[tt_sample])
            if np.all(np.isnan(spindle_peak_freq)):
                spindle_peak_freq[:] = 13.5
            else:
                spindle_peak_freq[np.isnan(spindle_peak_freq)] = np.nanmedian(spindle_peak_freq)
            sp_res = my_spindle_detect(
                eeg, sleep_stages, Fs, eeg_ch_names,
                include=[2,1],
                freq_sp=[[x-1,x+1] for x in spindle_peak_freq],
                thresh={'corr':0.73, 'rel_pow':0.07, 'rms':1},
                verbose=False)

        has_spindle_F = np.zeros(len(tt))
        has_spindle_C = np.zeros(len(tt))
        has_spindle_O = np.zeros(len(tt))
        for i in range(len(sp_res)):
            idx = np.searchsorted(tt, sp_res.Peak.iloc[i])-1
            chn = sp_res.Channel.iloc[i]
            if chn in eeg_ch_names_f:
                has_spindle_F[idx] = 1
            elif chn in eeg_ch_names_c:
                has_spindle_C[idx] = 1
            elif chn in eeg_ch_names_o:
                has_spindle_O[idx] = 1
            
        ## SWA amp and perc

        sw_amp_ptp_F = np.zeros(len(tt))
        sw_amp_ptp_C = np.zeros(len(tt))
        sw_amp_ptp_O = np.zeros(len(tt))
        for i in range(len(sw_res)):
            idx = np.searchsorted(tt, sw_res.MidCrossing.iloc[i])-1
            chn = sw_res.Channel.iloc[i]
            if chn in eeg_ch_names_f:
                sw_amp_ptp_F[idx] = max(sw_amp_ptp_F[idx], sw_res.PTP.iloc[i])
            elif chn in eeg_ch_names_c:
                sw_amp_ptp_C[idx] = max(sw_amp_ptp_C[idx], sw_res.PTP.iloc[i])
            elif chn in eeg_ch_names_o:
                sw_amp_ptp_O[idx] = max(sw_amp_ptp_O[idx], sw_res.PTP.iloc[i])
        sw_perc_F = np.zeros(len(tt))
        sw_perc_C = np.zeros(len(tt))
        sw_perc_O = np.zeros(len(tt))
        for i, start in enumerate(tt_sample):
            sw_perc_F[i] = sw_mask[[0,1], start:start+epoch_size].mean()  #TODO: [0,1]
            sw_perc_C[i] = sw_mask[[2,3], start:start+epoch_size].mean()
            sw_perc_O[i] = sw_mask[[4,5], start:start+epoch_size].mean()

        ## rapid eye movement

        has_rem = np.zeros(len(tt))
        for i in range(len(rem_res)):
            idx = np.searchsorted(tt, rem_res.Peak.iloc[i])-1
            has_rem[idx] = 1
        
        ## EMG

        env_seg = np.array([env[x:x+epoch_size] for x in tt_sample])
        env_seg = env_seg.mean(axis=1)

        df_feats[epoch_time] = pd.DataFrame(data={
            'alpha_rel_bp_F':alpha_rel_bp_F, 'alpha_rel_bp_C':alpha_rel_bp_C, 'alpha_rel_bp_O':alpha_rel_bp_O,
            'theta_rel_bp_F':theta_rel_bp_F, 'theta_rel_bp_C':theta_rel_bp_C, 'theta_rel_bp_O':theta_rel_bp_O,
            'has_spindle_F':has_spindle_F, 'has_spindle_C':has_spindle_C, 'has_spindle_O':has_spindle_O,
            'sw_amp_ptp_F':sw_amp_ptp_F, 'sw_amp_ptp_C':sw_amp_ptp_C, 'sw_amp_ptp_O':sw_amp_ptp_O,
            'sw_perc_F':sw_perc_F, 'sw_perc_C':sw_perc_C, 'sw_perc_O':sw_perc_O,
            'has_rem':has_rem, 'emg_env_rank_mean':env_seg,
            })
        epoch_start_idss[epoch_time] = epoch_start_ids
        sleep_stages_epochs[epoch_time] = sleep_stages_epoch
    
    if only_one_epoch_time:
        et = epoch_times[0]
        return df_feats[et], sleep_stages_epochs[et], epoch_start_idss[et], sp_res, sw_res, rem_res
    else:
        return df_feats, sleep_stages_epochs, epoch_start_idss, sp_res, sw_res, rem_res


if __name__=='__main__':
    outcome = 'Dementia'
    epoch_times = [30,15,10,5]
    psg_base_folder = '/bdsp/opendata/PSG/data/S0001'
    n_jobs = 16

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
    def _get_features(df_row):
        signals, sleep_stages, params = load_data(df_row.SignalPath.split('/')[-2], psg_base_folder, ch_names_re, df_annot=df_annot)
        signals = signals.values.T
        signals, sleep_stages = remove_bad_start_end(signals, sleep_stages)
        Fs = params['Fs']

        signals = np.concatenate([
            filter_signal(signals[:8], Fs, 60, [0.3, 35]),
            filter_signal(signals[8:], Fs, 60, [10, 100 if Fs/2>100 else None]),
        ], axis=0)

        #features, sleep_stages_epoch, epoch_start_ids = 
        return get_features(pd.DataFrame(data=signals.T, columns=ch_names), sleep_stages, Fs, epoch_times)

    with Parallel(n_jobs=n_jobs, verbose=False) as pp:
        result = pp(delayed(_get_features)(df.iloc[i]) for i in tqdm(range(len(df))))

    df_feat = {et:defaultdict(list) for et in epoch_times}
    sp_ress = []; sw_ress = []; rem_ress = []
    for i in range(len(df)):
        hashid = df.HashID.iloc[i]
        dov = df.DOVshifted.iloc[i]
        features, sleep_stages_epoch, epoch_start_ids, sp_res, sw_res, rem_res = result[i]

        sp_res.insert(0, 'DOV', dov); sp_res.insert(0, 'HashID', hashid); sp_ress.append(sp_res)
        sw_res.insert(0, 'DOV', dov); sw_res.insert(0, 'HashID', hashid); sw_ress.append(sw_res)
        rem_res.insert(0, 'DOV', dov); rem_res.insert(0, 'HashID', hashid); rem_ress.append(rem_res)

        for et in epoch_times:
            df_feat[et]['HashID'].extend([hashid]*len(features[et]))
            df_feat[et]['DOVshifted'].extend([dov]*len(features[et]))
            df_feat[et]['SleepStage'].extend(sleep_stages_epoch[et])
            df_feat[et]['EpochStartIdx'].extend(epoch_start_ids[et])
            for col in features[et].columns:
                df_feat[et][col].extend(features[et][col])

    save_paths = {et:f'features_epoch{et}s.csv.zip' for et in epoch_times}
    for et in epoch_times:
        pd.DataFrame(data=df_feat[et]).to_csv(save_paths[et], index=False, compression='zip')
    pd.concat(sp_ress, axis=0, ignore_index=True).to_csv('spindle_detection_result.csv', index=False)
    pd.concat(sw_ress, axis=0, ignore_index=True).to_csv('slow_wave_detection_result.csv', index=False)
    pd.concat(rem_ress, axis=0, ignore_index=True).to_csv('rem_detection_result.csv', index=False)

