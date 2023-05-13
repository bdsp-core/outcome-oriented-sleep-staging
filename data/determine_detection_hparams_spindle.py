import os
import pickle
import re
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, cohen_kappa_score
from skopt import gp_minimize
from skopt.utils import cook_initial_point_generator
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

    metric = 'f1'
    random_state = 2023
    
    # get eegs

    tmp_data_path = 'event_detection_tmp_eeg_data.pickle'
    if not os.path.exists(tmp_data_path):
        import mne
        mne.set_log_level('ERROR')
        from manual_label_detection import load_data, filter_signal
        from pattern_detection import get_spindle_peak_freq
    
        #folders = [x for x in os.listdir('.') if re.match('[0-9a-z]{64}_[0-9]{8}_[0-9]{9}',x)]
        folders = pd.read_excel('manual_check/manual_check_spindle.xlsx').SID.unique().astype(str)
        df_annot = pd.read_csv('annotations_sleep_stages.zip', compression='zip')
        df_annot['DOVshifted'] = pd.to_datetime(df_annot.DOVshifted)

        rel_pow = []
        mcorr = []
        mrms = []
        #eegs = []
        spindle_peak_freqs = []
        sleep_stages_mgh = []
        #base_folder = '.'
        base_folder = '/bdsp/opendata/PSG/data/S0001'
        for si, subject_folder in enumerate(tqdm(folders)):
            signals, sleep_stages, params = load_data(subject_folder, base_folder, ch_names_re, df_annot=df_annot)
            Fs = params['Fs']
            start_time = params['start_time']
            signals = signals.rename(columns={k:v for k,v in zip(params['channel_names'],ch_names)})
            sleep_stages[np.isnan(sleep_stages)] = -1
            sleep_stages = sleep_stages.astype(int)

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
        
    def loss_func(params, return_y_yp=False, use_ids=None):
        if use_ids is None:
            use_ids = range(len(folders))
        res = Parallel(n_jobs=16)(delayed(get_y_yp)(si, params) for si in use_ids)
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
        
    df_y = pd.read_excel('manual_check/manual_check_spindle.xlsx')
    df_y = df_y[np.in1d(df_y.Channel, eeg_ch_names)].reset_index(drop=True)
    
    # given a parameter set, generate detections
    """
    df0 = pd.read_excel('params_performances_spindle.xlsx')
    df0 = df0[
            (df0['corr']>=0.6)&(df0['corr']<=0.8)&\
            (df0['rel_pow']>=0)&(df0['rel_pow']<=0.2)&\
            (df0['rms']>=1)&(df0['rms']<=2)].reset_index(drop=True)
    x0 = df0.iloc[:,:3].values.tolist()
    y0 = (-df0.iloc[:,3].values).tolist()
    """

    Ncv = 5
    cv = StratifiedKFold(n_splits=Ncv, shuffle=True, random_state=random_state)
    sids_y = np.array([df_y.ManualOK[df_y.SID==sid].mean()>0.15 for sid in folders]).astype(int)
    _ = np.zeros((len(folders),1))
    loss_cv = []
    y_cv = []
    yp_cv = []
    param_cv = []
    for cvi, (ids_tr, ids_te) in enumerate(cv.split(_, sids_y)):
        print(f'\n================= CV = {cvi+1}/{Ncv} ==================\n')

        loss_func_ = lambda x:loss_func(x, return_y_yp=False, use_ids=ids_tr)
        opt_res = gp_minimize(loss_func_, [
            (0.6,0.8), # corr
            (0,0.4), # rel_pow
            (0.,3.),   # rms
            ],
            n_calls=50, n_initial_points=10,
            initial_point_generator=cook_initial_point_generator("lhs", criterion="maximin"),
            #x0=x0, y0=y0,
            random_state=random_state, verbose=10, callback=lambda x:print(x.x), n_jobs=1)
        loss, y, yp = loss_func(opt_res.x, return_y_yp=True, use_ids=ids_te)
        print(loss)
        loss_cv.append(loss)
        y_cv.append(y)
        yp_cv.append(yp)
        param_cv.append(opt_res.x)
    import pdb;pdb.set_trace()

    param_cv = np.array(param_cv)
    loss_cv = np.array(loss_cv)

    cm_cv = confusion_matrix(np.concatenate(y_cv), np.concatenate(yp_cv))
    f1_cv1 = f1_score(np.concatenate(y_cv), np.concatenate(yp_cv))
    f1_cv2 = np.mean([f1_score(y_cv[i], yp_cv[i]) for i in range(Ncv)])
    mcc_cv1 = matthews_corrcoef(np.concatenate(y_cv), np.concatenate(yp_cv))
    mcc_cv2 = np.mean([matthews_corrcoef(y_cv[i], yp_cv[i]) for i in range(Ncv)])
    cohenkappa_cv1 = cohen_kappa_score(np.concatenate(y_cv), np.concatenate(yp_cv))
    cohenkappa_cv2 = np.mean([cohen_kappa_score(y_cv[i], yp_cv[i]) for i in range(Ncv)])
    print('CV')
    print(np.c_[param_cv, loss_cv])
    print(cm_cv)
    print(f'f1  = {f1_cv1}, {f1_cv2}')
    print(f'mcc = {mcc_cv1}, {mcc_cv2}')
    print(f'k   = {cohenkappa_cv1}, {cohenkappa_cv2}')

    params_refit = np.sum(param_cv*loss_cv.reshape(-1,1),axis=0)/loss_cv.sum()#np.median(param_cv, axis=0)
    loss, y, yp = loss_func(params_refit, return_y_yp=True)
    cm = confusion_matrix(y, yp)
    f1 = f1_score(y, yp)
    mcc = matthews_corrcoef(y, yp)
    cohenkappa = cohen_kappa_score(y, yp)
    print('refit')
    print(params_refit)
    print(cm)
    print(f'f1  = {f1}')
    print(f'mcc = {mcc}')
    print(f'k   = {cohenkappa}')
    """
CV
array([[ 0.74267049,  0.1040685 ,  0.        , -0.42622951],
       [ 0.72839452,  0.06551942,  3.        , -0.73239437],
       [ 0.74286243,  0.06083125,  0.        , -0.36      ],
       [ 0.75142567,  0.06219886,  0.        , -0.46938776],
       [ 0.74266268,  0.04159757,  0.52412241, -0.48275862]])
array([[776, 112],
       [160, 179]])
f1  = 0.5682539682539683, 0.4941540500371201
mcc = 0.42251019065585405, 0.35372920942100333
k   = 0.4202934800733701, 0.3464660904957862

refit
array([0.74012847, 0.06618152, 0.99167768])
[[738, 150]
 [112, 227]]
f1  = 0.6340782122905029
mcc = 0.48529040418687525
k   = 0.4839290702266401

==> [0.74, 0.07, 1]
[[737, 151]
 [112, 227]]
f1  = 0.6331938633193863
mcc = 0.48384272880292256
k   = 0.4824154938048839
    """
    
    #df_res = pd.DataFrame(data=np.array(opt_res.x_iters), columns=['corr', 'rel_pow','rms'])
    #col = 'Performance_'+metric
    #df_res[col] = -opt_res.func_vals
    #df_res = df_res.sort_values(col, ignore_index=True, ascending=False)
    #print(df_res)
    #df_res.to_excel('params_performances_spindle.xlsx', index=False)



if __name__=='__main__':
    main()

