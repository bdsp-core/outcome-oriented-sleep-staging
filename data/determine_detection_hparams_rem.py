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
from pattern_detection import my_rem_detect


def main():
    ch_names = ['e1-m2', 'e2-m1']
    ch_names_re = ['e1-', 'e2-']

    metric = 'f1'
    random_state = 2023
    
    # get eegs

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
                'eogs':eogs, 'sids':folders, 'Fs':Fs,
                'sleep_stages_mgh':sleep_stages_mgh, 'channel_names':ch_names,
                }, ff)
    else:
        print(f'reading from {tmp_data_path}...')
        with open(tmp_data_path, 'rb') as ff:
            res = pickle.load(ff)
        eogs = res['eogs']
        folders = res['sids']
        Fs = res['Fs']
        sleep_stages_mgh = res['sleep_stages_mgh']

    def get_y_yp(si, params):
        subject_folder = folders[si]
        eog = eogs[si]
        sleep_stages = sleep_stages_mgh[si]

        res = my_rem_detect(eog[0], eog[1], sleep_stages, Fs, ch_names, include=[1,2,4],
            amplitude=[amp_lb, amp_lb+amp_int],
            duration=[dur_lb, dur_lb+dur_int],
            freq_rem=[0.5,5],
            verbose=False)
        #TODO

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
        
    df_y = pd.read_excel('manual_check/manual_check_rem.xlsx')
    df_y = df_y[np.in1d(df_y.Channel, ch_names)].reset_index(drop=True)
    
    # given a parameter set, generate detections

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
            (),  #TODO amp_lb
            (),  # amp_int
            (),  # dur_lb
            (),  # dur_int
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
    """


if __name__=='__main__':
    main()

