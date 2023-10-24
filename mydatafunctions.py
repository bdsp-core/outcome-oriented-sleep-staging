import re
import datetime
import numpy as np
import pandas as pd
from mne_bids import read_raw_bids, BIDSPath, find_matching_paths#, make_report


def read_dataset_mgh_bids(root_path, sid, dov):
    site = 'S0001'
    sid = site+str(sid)

    # find session
    paths = find_matching_paths(root_path, subjects=sid, extensions='_scans.tsv')
    dovs = [datetime.datetime.strptime(pd.read_csv(x.fpath, sep='\t').acq_time.iloc[0], '%Y-%m-%dT%H:%M:%S.%fZ').date() for x in paths]
    session = paths[dovs.index(dov.date())].session
    path = BIDSPath(root=root_path, subject=sid, datatype='eeg', task='psg', session=session)

    # read data in BIDS format
    edf = read_raw_bids(bids_path=path, verbose=False)
    Fs = edf.info['sfreq']
    start_time = edf.info['meas_date'].replace(tzinfo=None)

    #df_annot = pd.DataFrame(edf.annotations)
    #desc_col = 'description'
    desc_col = 'value'
    fpath_old = str(path.fpath).replace('_eeg.edf', '_annotations.csv')
    df_annot = pd.read_csv(fpath_old)
    df_annot['onset'] = (pd.to_datetime(df_annot.time)-start_time).apply(lambda x:x.seconds)
    df_annot.loc[pd.isna(df_annot.duration), 'duration'] = 1
    df_annot = df_annot.rename(columns={'event':desc_col})
    df_annot = df_annot[['onset', 'duration', desc_col]]
    #fpath_new = str(path.fpath).replace('_eeg.edf', '_events.tsv')
    #df_annot.to_csv(fpath_new, sep='\t', index=False)

    ch_names = [
        'f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1',
        'e1-m2', 'chin1-chin2', 'abd', 'chest', 'ecg']
    ch_names_re = [
        'f3-m', 'f4-m', 'c3-m', 'c4-m', 'o1-m', 'o2-m',
        'e1-m', 'chin1-chin2', 'abd', 'chest', 'e[ck]g']
    ch_names_data = [[x for x in edf.ch_names if re.match(y,x,re.I)] for y in ch_names_re]
    assert all([len(x)==1 for x in ch_names_data]), (sid,dov,ch_names_data,edf.ch_names)
    ch_names_data = [x[0] for x in ch_names_data]
    signals = edf.get_data(picks=ch_names_data)
    signals *= 1e6
    #TODO reverse ECG?
    signals = pd.DataFrame(signals.T, columns=ch_names)

    sleep_stages = np.zeros(len(signals))+np.nan
    mapping = {
            'n4':1, 'nrem4':1, 'nrem 4':1, 's4':1,
            'n3':1, 'nrem3':1, 'nrem 3':1, 's3':1,
            'n2':2, 'nrem2':2, 'nrem 2':2, 's2':2,
            'n1':3, 'nrem1':3, 'nrem 1':3, 's1':3,
            'r':4, 'rem':4,
            'w':5, 'wake':5, 'awake':5, }
    for i in range(len(df_annot)):
        desc = str(df_annot.loc[i,desc_col]).lower()
        if not desc.startswith('sleep_stage_'):
            continue
        s = desc.split('_')[-1]
        if s in mapping:
            start = int((df_annot.onset.iloc[i]*Fs))
            end   = int(((df_annot.onset.iloc[i]+df_annot.duration.iloc[i])*Fs))
            if not (0<=start<end<len(sleep_stages)):
                continue
            sleep_stages[start:end] = mapping[s]

    params = {'start_time':start_time, 'Fs':Fs}#'ch_names':ch_names,
    return signals, sleep_stages, params

