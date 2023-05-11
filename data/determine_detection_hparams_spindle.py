#from collections import defaultdict
#from itertools import product
import os
import pickle
import re
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, cohen_kappa_score
from skopt import gp_minimize
import sys
sys.path.insert(0, '..')
from pattern_detection import my_spindle_detect


def main():
    eeg_ch_names = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1']#, 'o1-m2', 'o2-m1']
    #eog_ch_names = ['e1-m2', 'e2-m1']
    ch_names = eeg_ch_names# + eog_ch_names
    eeg_ch_names_re = ['f3-', 'f4-', 'c3-', 'c4-']#, 'o1-', 'o2-']
    #eog_ch_names_re = ['e1-', 'e2-']
    ch_names_re = eeg_ch_names_re# + eog_ch_names_re
    
    df_y = pd.read_excel('manual_check/manual_check_spindle.xlsx')
    df_y = df_y[np.in1d(df_y.Channel, eeg_ch_names)].reset_index(drop=True)
    
    # get eegs

    tmp_data_path = 'event_detection_tmp_data.pickle'
    if not os.path.exists(tmp_data_path):
        import mne
        from manual_label_detection import load_data, filter_signal
        from pattern_detection import get_spindle_peak_freq
    
        folders = [x for x in os.listdir('.') if re.match('[0-9a-z]{64}_[0-9]{8}_[0-9]{9}',x)]
        df_annot = pd.read_csv('annotations_sleep_stages.zip', compression='zip')
        df_annot['DOVshifted'] = pd.to_datetime(df_annot.DOVshifted)

        rel_pow = []
        mcorr = []
        mrms = []
        #eegs = []
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

            #if pattern=='spindle':
            epoch_size = int(round(30*Fs))
            start_ids = np.arange(0, len(signals)-epoch_size+1, epoch_size)
            sleep_stages_epochs = sleep_stages[start_ids]
            epochs = np.array([eeg[:,x:x+epoch_size] for x in start_ids])
            spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs, fmin=0.3, fmax=35, bandwidth=0.5, adaptive=False, low_bias=True, normalization='full', verbose=False)
            spindle_peak_freq = get_spindle_peak_freq(spec, freq, sleep_stages_epochs)
            print(spindle_peak_freq)
            spindle_peak_freq[np.isnan(spindle_peak_freq)] = np.nanmedian(spindle_peak_freq)

            _, rel_pow_, mcorr_, mrms_ = my_spindle_detect(
                eeg, sleep_stages, Fs, eeg_ch_names, include=[2],
                freq_sp=[[spindle_peak_freq[chi]-1,spindle_peak_freq[chi]+1] for chi in range(len(eeg))],
                thresh={'corr':0.65, 'rel_pow':0.2, 'rms':1.5, 'amp':150},
                return_precomputed=True, compute_sp_char=False)
            
            rel_pow.append(rel_pow_)
            mcorr.append(mcorr_)
            mrms.append(mrms_)
            #eegs.append( eeg )
            spindle_peak_freqs.append(spindle_peak_freq)
            sleep_stages_mgh.append(sleep_stages)

        with open(tmp_data_path, 'wb') as ff:
            pickle.dump({
                'rel_pow':rel_pow, 'mcorr':mcorr, 'mrms':mrms, #'eegs':eegs,
                'spindle_peak_freqs':spindle_peak_freqs,
                'sids':folders, 'Fs':Fs,
                'sleep_stages_mgh':sleep_stages_mgh, 'channel_names':eeg_ch_names,
                }, ff)
    else:
        print(f'reading from {tmp_data_path}...')
        with open(tmp_data_path, 'rb') as ff:
            res = pickle.load(ff)
        rel_pow = res['rel_pow']
        mcorr = res['mcorr']
        mrms = res['mrms']
        #eegs = res['eegs']
        spindle_peak_freqs = res['spindle_peak_freqs']
        folders = res['sids']
        Fs = res['Fs']
        sleep_stages_mgh = res['sleep_stages_mgh']
        eeg_ch_names = res['channel_names']


    metric = 'mcc'
    
    def get_y_yp(si, params):
        subject_folder = folders[si]
        rel_pow_ = rel_pow[si]
        mcorr_ = mcorr[si]
        mrms_ = mrms[si]
        #eeg = eegs[si]
        spindle_peak_freq = spindle_peak_freqs[si]
        sleep_stages = sleep_stages_mgh[si]

        _ = np.random.randn(len(eeg_ch_names),len(sleep_stages))*100
        detection_mask = np.zeros((len(eeg_ch_names),len(sleep_stages)), dtype=bool)
        res = my_spindle_detect(_, sleep_stages, Fs, eeg_ch_names, include=[2],
            freq_sp=[[spindle_peak_freq[chi]-1,spindle_peak_freq[chi]+1] for chi in range(len(eeg_ch_names))],
            thresh={'corr':params[0], 'rel_pow':params[1], 'rms':params[2]}, verbose=False,
            rel_pow_all=rel_pow_, mcorr_all=mcorr_, mrms_all=mrms_,
            return_precomputed=False, compute_sp_char=False)

        for ri in range(len(res)):
            start_idx = int(round(res.Start[ri]*Fs))
            end_idx = int(round(res.End[ri]*Fs))
            chi = res.IdxChannel[ri]
            detection_mask[chi,start_idx:end_idx] = True

        yp = []
        y = []
        df_y_ = df_y[df_y.SID==subject_folder].reset_index(drop=True)
        for yi in range(len(df_y_)):
            start_idx = int(round(df_y_.Start[yi]*Fs))
            end_idx = int(round(df_y_.End[yi]*Fs))
            chi = eeg_ch_names.index(df_y_.Channel[yi])
            yp.append( int(detection_mask[chi,start_idx:end_idx].any()) )
        y.extend(df_y_.ManualOK)
        
        #res = res[np.in1d(res.Stage, [3,4,5])].reset_index(drop=True)
        #yp.extend([1]*len(res))
        #y.extend([0]*len(res))
        
        return y, yp
        
    def loss_func(params, return_y_yp=False):
        res = Parallel(n_jobs=12)(delayed(get_y_yp)(si, params) for si in range(len(folders)))
        y = np.concatenate([x[0] for x in res])
        yp = np.concatenate([x[1] for x in res])
                
        # compute metrics
        if metric=='f1':
            perf = f1_score(y,yp)
        elif metric=='mcc':
            perf = matthews_corrcoef(y,yp)
        elif metric=='kappa':
            perf = cohen_kappa_score(y,yp)
        if return_y_yp:
            return -perf, y, yp
        else:
            return -perf
        
    # given a parameter set, generate detections
    random_state = 2023
    opt_res = gp_minimize(loss_func, [
        (0.5,0.8), # corr
        (0.1,0.4), # rel_pow
        (1.2,3),   # rms
        ], n_calls=200, n_initial_points=20, initial_point_generator='lhs', random_state=random_state, verbose=10, callback=None, n_jobs=1)
    import pdb;pdb.set_trace()
    loss, y, yp = loss_func(opt_res.x, return_y_yp=True)
    #opt_res.x
    #[0.7827770696095304, 0.1051900672042389, 3.0]
    #opt_res.fun
    #-0.15461071231363888

    cm = confusion_matrix(y, yp)
    f1 = f1_score(y,yp)
    mcc = matthews_corrcoef(y,yp)
    cohenkappa = cohen_kappa_score(y,yp)
    print(opt_res.x)
    print(cm)
    print(f'f1  = {f1}')
    print(f'mcc = {mcc}')
    print(f'k   = {cohenkappa}')
    
    df_res = pd.DataFrame(data=np.array(opt_res.x_iters), columns=['corr', 'rel_pow','rms'])
    col = 'Performance_'+metric
    df_res[col] = -opt_res.func_vals
    df_res = df_res.sort_values(column=col)
    print(df_res)
    df_res.to_excel('params_performances_spindle.xlsx', index=False)



if __name__=='__main__':
    main()

