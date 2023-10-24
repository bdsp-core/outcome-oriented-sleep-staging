import numpy as np
import pandas as pd
from mne.filter import filter_data, notch_filter
from mne.time_frequency import psd_array_multitaper
from scipy.signal import resample
from sklearn.preprocessing import RobustScaler


def clip_noisy_values(psg, sample_rate, period_length_sec,
                      min_max_times_global_iqr=20):
    """
    Clips all values that are larger or smaller than +- min_max_times_global_iqr
    times to IQR of the whole channel.
    Args:
        psg:                      A ndarray of shape [N, C] of PSG data
        sample_rate:              The sample rate of data in the PSG
        period_length_sec:        The length of one epoch/period/segment in
                                  seconds
        min_max_times_global_iqr: Extreme value threshold; number of times a
                                  value in a channel must exceed the global IQR
                                  for that channel for it to be termed an
                                  outlier (in neg. or pos. direction).
    Returns:
        PSG, ndarray of shape [N, C]
        A list of lists, one sub-list for each channel, each storing indices
        of all epochs in which one or more values were clipped.
    """
    n_channels = psg.shape[-1]
    chan_inds = []
    for chan in range(n_channels):
        chan_psg = psg[..., chan]

        # Compute global IQR
        iqr = np.subtract(*np.percentile(chan_psg, [75, 25]))
        threshold = iqr * min_max_times_global_iqr

        # Reshape PSG to periods on 0th axis
        n_periods = int(chan_psg.shape[0]/(sample_rate*period_length_sec))
        temp_psg = chan_psg.reshape(n_periods, -1)

        # Compute IQR for all epochs
        inds = np.unique(np.where(np.abs(temp_psg) > threshold)[0])
        chan_inds.append(inds)

        # Zero out noisy epochs in the particular channel
        psg[:, chan] = np.clip(chan_psg, -threshold, threshold)
    return psg, chan_inds


def preprocess(signals, Fs, sleep_stages):
    notch_freq_us = 60.                 # [Hz]
    notch_freq_eur = 50.                # [Hz]
    #bandpass_freq_eeg = [0.1, 20]       # [Hz]
    bandpass_freq_eeg = [0.3, 35]       # [Hz]
    bandpass_freq_airflow = [0., 10]    # [Hz]
    bandpass_freq_ecg = [0.3, None]     # [Hz]
    new_Fs = 100                        # [Hz]
    period_length_sec = 0.5 # second
    epoch_time = 30  # second

    epoch_size = int(round(epoch_time*Fs))
    start_ids = np.arange(0, len(sleep_stages)-epoch_size+1, epoch_size)
    sleep_stages2 = sleep_stages[start_ids+epoch_size//2]

    X = signals.values.T
    # 1. Notch filter
    X = notch_filter(X, Fs, notch_freq_us, verbose=False)

    # 2. Bandpass filter
    ids = np.in1d(signals.columns, ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2'])
    X[ids] = filter_data(X[ids], Fs, bandpass_freq_eeg[0], bandpass_freq_eeg[1], verbose=False)
    ids = np.in1d(signals.columns, ['abd', 'chest', 'airflow', 'ptaf', 'cflow'])
    X[ids] = filter_data(X[ids], Fs, bandpass_freq_airflow[0], bandpass_freq_airflow[1], verbose=False)
    ids = np.in1d(signals.columns, ['ecg'])
    X[ids] = filter_data(X[ids], Fs, bandpass_freq_ecg[0], bandpass_freq_ecg[1], verbose=False)

    # 3. Resample data
    if new_Fs != Fs:
        X = resample(X, int(round(X.shape[1]/Fs*new_Fs)), axis=-1)
        Fs = new_Fs

    # 4. scale
    #X, chan_inds = clip_noisy_values(X.T, Fs, period_length_sec, min_max_times_global_iqr=20)
    transformer = RobustScaler().fit(X.T)

    # 5. segment
    epoch_size = int(round(epoch_time*Fs))
    start_ids = np.arange(0, X.shape[1]-epoch_size+1, epoch_size)
    segs = np.array([X[:,x:x+epoch_size] for x in start_ids])

    spec, freq = psd_array_multitaper(segs, Fs, fmin=0, fmax=np.inf, bandwidth=0.5, normalization='full', verbose=False)

    segs = (segs-transformer.center_.reshape(1,-1,1))/transformer.scale_.reshape(1,-1,1)
    
    return segs, sleep_stages2, spec, freq, Fs

