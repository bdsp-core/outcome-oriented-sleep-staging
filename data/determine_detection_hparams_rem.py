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
from pattern_detection import my_sw_detect


def main():
    ch_names = ['e1-m2', 'e2-m1']
    ch_names_re = ['e1-', 'e2-']
    
    # get eogs

    tmp_data_path = 'event_detection_tmp_eog_data.pickle'
    if not os.path.exists(tmp_data_path):
        import mne
        mne.set_log_level('ERROR')
        from manual_label_detection import load_data, filter_signal
        from pattern_detection import get_spindle_peak_freq
    
        #folders = [x for x in os.listdir('.') if re.match('[0-9a-z]{64}_[0-9]{8}_[0-9]{9}',x)]
        folders = pd.read_excel('manual_check/manual_check_spindle.xlsx').SID.unique().astype(str)
        df_annot = pd.read_csv('annotations_sleep_stages.zip', compression='zip')
        df_annot['DOVshifted'] = pd.to_datetime(df_annot.DOVshifted)

        eogs = []
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

            eog = signals[ch_names].values.T
            eog = filter_signal(eog, Fs, 60, [0.3, 35])

            eogs.append( eog )
            sleep_stages_mgh.append(sleep_stages)

        with open(tmp_data_path, 'wb') as ff:
            pickle.dump({
                'eogs':eogs, 'sleep_stages_mgh':sleep_stages_mgh,
                'sids':folders, 'Fs':Fs,
                'channel_names':ch_names,
                }, ff)
    else:
        print(f'reading from {tmp_data_path}...')
        with open(tmp_data_path, 'rb') as ff:
            res = pickle.load(ff)
        eogs = res['eogs']
        folders = res['sids']
        Fs = res['Fs']
        sleep_stages_mgh = res['sleep_stages_mgh']


    metric = 'f1'
    
    def get_y_yp(si, params):
        subject_folder = folders[si]
        eog = eogs[si]
        sleep_stages = sleep_stages_mgh[si]

        amp_lb, amp_int, dur_lb, dur_int = params
        amp_ub = amp_lb+amp_int
        dur_ub = dur_lb+dur_int

        res = my_sw_detect(eog[0], eog[1], sleep_stages, Fs, ch_names, include=[1,2,4],
            amplitude=[amp_lb, amp_ub], duration=[dur_lb, dur_ub], freq_rem=[0.5,5], verbose=False)
        #TODO

        return y, yp
        
    def loss_func(params, return_y_yp=False):
        res = Parallel(n_jobs=16)(delayed(get_y_yp)(si, params) for si in range(len(folders)))
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
    """
    df0 = pd.read_excel('params_performances_spindle.xlsx')
    df0 = df0[
            (df0['corr']>=0.6)&(df0['corr']<=0.8)&\
            (df0['rel_pow']>=0)&(df0['rel_pow']<=0.2)&\
            (df0['rms']>=1)&(df0['rms']<=2)].reset_index(drop=True)
    x0 = df0.iloc[:,:3].values.tolist()
    y0 = (-df0.iloc[:,3].values).tolist()
    """

    #TODO cross validation
    random_state = 2023
    opt_res = gp_minimize(loss_func, [
        (30,80), # amp_lb
        (150,350), # amp_int
        (0.1,1),   # dur_lb
        (0.4,1),   # dur_int
        ],
        n_calls=100, n_initial_points=20, initial_point_generator='lhs',
        #x0=x0, y0=y0,
        random_state=random_state, verbose=10, callback=None, n_jobs=1)
    loss, y, yp = loss_func(opt_res.x, return_y_yp=True)
    """
    """

    cm = confusion_matrix(y, yp)
    f1 = f1_score(y,yp)
    mcc = matthews_corrcoef(y,yp)
    cohenkappa = cohen_kappa_score(y,yp)
    print(opt_res.x)
    print(cm)
    print(f'f1  = {f1}')
    print(f'mcc = {mcc}')
    print(f'k   = {cohenkappa}')
    
    df_res = pd.DataFrame(data=np.array(opt_res.x_iters), columns=['amp_lb', 'amp_int','dur_lb', 'dur_int'])
    col = 'Performance_'+metric
    df_res[col] = -opt_res.func_vals
    df_res = df_res.sort_values(col, ignore_index=True, ascending=False)
    print(df_res)
    df_res.to_excel('params_performances_rem.xlsx', index=False)



if __name__=='__main__':
    main()

