import datetime
import os
import mne
import sys
#sys.path.insert(0, '/home/ubuntu/sleep_general')
sys.path.insert(0, r'D:\projects\sleep_general')
from mgh_sleeplab import load_mgh_signal, annotations_preprocess, vectorize_sleep_stages


def load_data(subject_folder, base_folder, ch_names, df_annot=None):
    # load signal
    signal_path = os.path.join(base_folder, subject_folder, 'Shifted_Signal_'+subject_folder+'.mat')
    signals, params = load_mgh_signal(signal_path, channels=ch_names)
    # load annotation
    hashid, dov1, dov2 = subject_folder.split('_')
    dov1 = datetime.datetime.strptime(dov1, '%Y%m%d')
    annot = df_annot[(df_annot.HashID==hashid)&(df_annot.DOVshifted==dov1)].reset_index(drop=True)
    annot = annotations_preprocess(annot, params['Fs'], verbose=False)
    sleep_stages = vectorize_sleep_stages(annot, len(signals))
    return signals, sleep_stages, params


def filter_signal(signals, Fs, notch_freq, bandpass_freq):
    signals = signals - signals.mean(axis=1, keepdims=True)
    signals = mne.filter.notch_filter(signals, Fs, notch_freq, verbose=False)
    signals = mne.filter.filter_data(signals, Fs, bandpass_freq[0], bandpass_freq[1], verbose=False)
    return signals
