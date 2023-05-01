from collections import defaultdict
from itertools import product
import os
import pickle
import re
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import mne
from tqdm import tqdm
import yasa
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, cohen_kappa_score
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
import seaborn as sns
sns.set_style('ticks')
from check_detection import load_data, filter_signal, get_spindle_peak_freq


def my_yasa_spindles_detect(data, sf=None, ch_names=None, hypno=None, include=[1,2,3], freq_sp=[12,15], freq_broad=[1,30], duration=[0.5,2], min_distance=500, thresh={'corr': 0.65, 'rel_pow': 0.2, 'rms': 1.5}, multi_only=False, remove_outliers=False, verbose=False):
    max_amp = thresh.pop('amp')
    t_standout = thresh.pop('t_standout')
    #f_standout = thresh.pop('f_standout')

    res = yasa.spindles_detect(data, sf=sf, ch_names=ch_names, hypno=hypno, include=include,
            freq_sp=freq_sp, freq_broad=freq_broad, duration=duration, min_distance=min_distance,
            thresh=thresh, multi_only=multi_only, remove_outliers=remove_outliers, verbose=verbose)

    cols = ['Start', 'Peak', 'End', 'Duration', 'Amplitude', 'RMS',
       'AbsPower', 'RelPower', 'Frequency', 'Oscillations', 'Symmetry',
       'Stage', 'Channel', 'IdxChannel']
    res = pd.DataFrame(columns=cols) if res is None else res.summary()
    res = res[res.Amplitude<max_amp].reset_index(drop=True)
    res['t_standout'] = np.nan

    # add t_standout and f_standout
    pad = int(round(2*sf))
    for ri in range(len(res)):
        start_time = res.Start[ri]
        end_time = res.End[ri]
        ch_idx = res.IdxChannel[ri]
        peak_freq = res.Frequency[ri]

        start_idx = int(round(start_time*sf))
        end_idx = int(round(end_time*sf))
        middle_time = (start_time+end_time)/2
        start_idx2 = int(round((middle_time-10)*sf))
        end_idx2 = int(round((middle_time+10)*sf))
        if start_idx2<0 or end_idx2>=data.shape[1]:
            res.loc[ri, 't_standout'] = np.nan
            continue
        data2 = data[ch_idx][start_idx2:end_idx2]
        data_f = mne.filter.filter_data(data2, sf, peak_freq-0.5, peak_freq+0.5, verbose=False)
        env_f = np.abs(hilbert(data_f))

        left = env_f[start_idx-start_idx2-pad:start_idx-start_idx2].mean()
        right = env_f[end_idx-start_idx2:end_idx-start_idx2+pad].mean()
        middle = env_f[start_idx-start_idx2:end_idx-start_idx2].max()
        res.loc[ri, 't_standout'] = middle/max(left,right)

    res = res[res.t_standout>=t_standout].reset_index(drop=True)
    #res = res[res.f_standout>=f_standout].reset_index(drop=True)

    return res


def main(pattern):
    df_y = pd.read_excel(f'manual_check/manual_check_{pattern}.xlsx')

    eeg_ch_names = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1']
    #eog_ch_names = ['e1-m2', 'e2-m1']
    ch_names = eeg_ch_names# + eog_ch_names
    eeg_ch_names_re = ['f3-', 'f4-', 'c3-', 'c4-', 'o1-', 'o2-']
    #eog_ch_names_re = ['e1-', 'e2-']
    ch_names_re = eeg_ch_names_re# + eog_ch_names_re

    # get eegs

    tmp_data_path = 'event_detection_tmp_data.pickle'
    if not os.path.exists(tmp_data_path):
        folders = [x for x in os.listdir('.') if re.match('[0-9a-z]{64}_[0-9]{8}_[0-9]{9}',x)]
        df_annot = pd.read_csv('annotations_sleep_stages.zip', compression='zip')
        df_annot['DOVshifted'] = pd.to_datetime(df_annot.DOVshifted)

        eegs = []
        spindle_peak_freqs = []
        sleep_stages_mgh = []
        for si, subject_folder in enumerate(tqdm(folders)):
            signals, sleep_stages, params = load_data(subject_folder, '.', ch_names_re, df_annot=df_annot)
            Fs = params['Fs']
            start_time = params['start_time']
            signals = signals.rename(columns={k:v for k,v in zip(params['channel_names'],ch_names)})

            eeg = signals[eeg_ch_names].values.T
            #eog = signals[eog_ch_names].values.T
            eeg = filter_signal(eeg, Fs, 60, [0.3, 35])
            #eog = filter_signal(eog, Fs, 60, [0.3, 35])


            if pattern=='spindle':
                spindle_peak_freq = get_spindle_peak_freq(eeg, sleep_stages, Fs)
                print(spindle_peak_freq)
                spindle_peak_freq[np.isnan(spindle_peak_freq)] = np.nanmedian(spindle_peak_freq)

            eegs.append( eeg )
            spindle_peak_freqs.append(spindle_peak_freq)
            sleep_stages_mgh.append(sleep_stages)

        with open(tmp_data_path, 'wb') as ff:
            pickle.dump({
                'eegs':eegs, 'spindle_peak_freqs':spindle_peak_freqs,
                'sids':folders, 'Fs':Fs,
                'sleep_stages_mgh':sleep_stages_mgh, 'channel_names':eeg_ch_names,
                }, ff)
    else:
        print(f'reading from {tmp_data_path}...')
        with open(tmp_data_path, 'rb') as ff:
            res = pickle.load(ff)
        eegs = res['eegs']
        spindle_peak_freqs = res['spindle_peak_freqs']
        folders = res['sids']
        Fs = res['Fs']
        sleep_stages_mgh = res['sleep_stages_mgh']
        eeg_ch_names = res['channel_names']

    # given a parameter set, generate detections
    params = {'corr':[0.6,0.65,0.7],
            'rel_pow':[0.15, 0.2, 0.25],
            'rms':[1.5,2,2.5,3],
            'max_amp':[80],
            'temporal_standout':[1.5,2,2.5,3],
            #'freq_standout':[2,3,4,5],
            }

    keys, values = zip(*params.items())
    len_ = np.prod([len(x) for x in values])
    df_res = defaultdict(list)
    for bundle in tqdm(product(*values), total=len_):
        param = dict(zip(keys, bundle))

        y_ = []
        yp_ = []
        for si, subject_folder in enumerate(tqdm(folders)):
            eeg = eegs[si]
            spindle_peak_freq = spindle_peak_freqs[si]
            sleep_stages = sleep_stages_mgh[si]

            detection_mask = np.zeros_like(eeg, dtype=bool)
            for chi in range(4):#len(eeg)):
                res = my_yasa_spindles_detect( eeg[[chi]], sf=Fs, ch_names=[eeg_ch_names[chi]],
                    hypno=sleep_stages, include=[2], freq_sp=[spindle_peak_freq[chi]-1,spindle_peak_freq[chi]+1], freq_broad=[1,30],
                    duration=[0.5,2], min_distance=500,
                    thresh={'corr':param['corr'], 'rel_pow':param['rel_pow'], 'rms':param['rms'],
                        'amp':param['max_amp'], 't_standout':param['temporal_standout']},#, 'f_standout':param['freq_standout']},
                    multi_only=False, remove_outliers=False, verbose=False)

                for ri in range(len(res)):
                    start_idx = int(round(res.Start[ri]*Fs))
                    end_idx = int(round(res.End[ri]*Fs))
                    detection_mask[chi,start_idx:end_idx] = True

                df_y_ = df_y[(df_y.SID==subject_folder)&(df_y.Channel==eeg_ch_names[chi])].reset_index(drop=True)
                for yi in range(len(df_y_)):
                    start_idx = int(round(df_y_.Start[yi]*Fs))
                    end_idx = int(round(df_y_.End[yi]*Fs))
                    yp_.append( int(detection_mask[chi,start_idx:end_idx].any()) )
                y_.extend(df_y_.ManualOK)
        # compute metrics
        cm = confusion_matrix(y_, yp_)
        f1 = f1_score(y_,yp_)
        mcc = matthews_corrcoef(y_,yp_)
        cohenkappa = cohen_kappa_score(y_,yp_)
        print(param)
        print(cm)
        print(f'f1  = {f1}')
        print(f'mcc = {mcc}')
        print(f'k   = {cohenkappa}')
        for k in param:
            df_res[k].append(param[k])
        df_res['y'].append(y_)
        df_res['yp'].append(yp_)
        df_res['confusion_matrix'].append(cm)
        df_res['F1'].append(f1)
        df_res['MCC'].append(mcc)
        df_res['CohenKappa'].append(cohenkappa)
        df_res_ = pd.DataFrame(df_res)
        #print(df_res_)
        df_res_.to_excel('params_performances.xlsx', index=False)
    #import pdb;pdb.set_trace()



if __name__=='__main__':
    import sys
    pattern = sys.argv[1].lower().strip()
    assert pattern in ['spindle', 'slow_wave', 'rem']
    main(pattern)

