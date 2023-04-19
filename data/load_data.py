import os
import sys
sys.path.insert(0, '../../sleep_general')
from mgh_sleeplab import *


def load_data(folder):
    """
    """
    sid, date, time = os.path.basename(folder).split('_')

    signal_path = os.path.join(folder, f'Shifted_Signal_{sid}_{date}_{time}.mat')
    signals, params = load_mgh_signal(signal_path)
    t0 = params['start_time']
    Fs = params['Fs']
    signals['t_second'] = np.arange(len(signals))/Fs

    annot_path = os.path.join(folder, f'{sid}_{date}_{time}_annotations.csv')
    df_annot = annotations_preprocess(pd.read_csv(annot_path), Fs, t0=t0)

    signals['stage'] = vectorize_sleep_stages(df_annot, len(signals), noscore_fill=-1)
    params['stage_code'] = {-1:'?',1:'3',2:'2',3:'1',4:'R',5:'W'}
    params['stage_txt2code'] = {v:k for k,v in params['stage_code'].items()}
    params['stage_color'] = {
            -1:(255/255,255/255,255/255),1:(3/255,203/255,255/255),
            2:(2/255,128/255,2/255),3:(255/255,255/255,0/255),
            4:(255/255,102/255,155/255),5:(150/255,150/255,150/255)}

    signals['arousal'] = vectorize_arousals(df_annot, len(signals))
    params['arousal_code'] = {1:'arousal'}
    params['arousal_color'] = {1:(255/255,0,255/255)}

    signals['resp'] = vectorize_respiratory_events(df_annot, len(signals))
    params['resp_code'] = { 1:'obstructive', 2:'central',
                            3:'mixed', 4:'hypopnea', 5:'rera'}
    params['resp_color'] = {1:(255/255,76/255,1/255), 2:(255/255,76/255,1/255),
                            3:(255/255,76/255,1/255), 4:(255/255,76/255,1/255), 5:(255/255,76/255,1/255)}

    signals['limb'] = vectorize_limb_movements(df_annot, len(signals))
    params['limb_code'] = { 1:'isolated', 2:'periodic',
                            3:'arousal', 4:'limb'}
    params['limb_color'] = {1:(0,0,0), 2:(0,0,0), 3:(0,0,0), 4:(0,0,0)}

    signals['body_position'] = vectorize_body_position(df_annot, len(signals), Fs)
    params['body_position_code'] = {0:'', 1:'S', 2:'L',
                                3:'R', 4:'P',
                                5:'', 6:''}
    params['body_position_color'] = { 1:(239/255,129/255,238/255), 2:(3/255,254/255,255/255),
                                3:(2/255,128/255,2/255), 4:(255/255, 165/255, 0/255),
                                5:(255/255,255/255,255/255), 6:(255/255,255/255,255/255)}

    return signals, params

