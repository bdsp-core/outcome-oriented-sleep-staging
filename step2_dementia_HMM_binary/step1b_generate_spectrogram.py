import os
import pickle
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
import mne
import sys
sys.path.insert(0, '..')
from myfunctions import load_data, filter_signal
from step1_generate_dataset import remove_bad_start_end


def main():
    outcome = 'Dementia'
    epoch_times = [30,15,10,5]
    psg_base_folder = '/bdsp/opendata/PSG/data/S0001'
    n_jobs = 16

    df = pd.read_csv(f'../data/mastersheet_matched_{outcome}.csv')
    df['DOVshifted'] = pd.to_datetime(df.DOVshifted)
    
    df_annot = pd.read_csv('../data/annotations_sleep_stages.zip', compression='zip')
    df_annot['DOVshifted'] = pd.to_datetime(df_annot.DOVshifted)

    eeg_ch_names = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']
    #eog_ch_names = ['EOGL', 'EOGR']
    #emg_ch_names = ['EMG']
    ch_names = eeg_ch_names# + eog_ch_names + emg_ch_names
    ch_names_re = ['F3-', 'F4-', 'C3-', 'C4-', 'O1-', 'O2-']#, 'E1-', 'E2-', 'chin']

    # multitaper bandwidth
    BW = {30:2/3, 15:2/3, 10:1, 5:1.2}
    NW = {k:v*k/2 for k,v in BW.items()}
    Nt = {k:int(v*2-1) for k,v in NW.items()}
    print(f'BW = {BW}')
    print(f'Ntaper = {Nt}')
    print(f'NW = {NW}')
    
    def _get_eeg_spectrogram(df_row):
        signals, sleep_stages, params = load_data(df_row.SignalPath.split('/')[-2], psg_base_folder, ch_names_re, df_annot=df_annot)
        signals = signals.values.T
        signals, sleep_stages = remove_bad_start_end(signals, sleep_stages)
        Fs = params['Fs']
        start_time = params['start_time']

        signals = filter_signal(signals, Fs, 60, [0.3, 35])

        stages = {}
        specs = {}
        freqs = {}
        epoch_start_sec = {}
        for et in epoch_times:
            epoch_size = int(round(et*Fs))
            start_ids = np.arange(0, signals.shape[1]-epoch_size+1, epoch_size)
            epochs = np.array([signals[:, x:x+epoch_size] for x in start_ids])
            stages[et] = sleep_stages[start_ids+epoch_size//2]

            epoch_start_sec[et] = start_ids/Fs
            specs[et], freqs[et] = mne.time_frequency.psd_array_multitaper(
                    epochs, Fs, fmin=0.3, fmax=20,
                    bandwidth=BW[et], normalization='full',
                    verbose=False)

        return stages, specs, freqs, epoch_start_sec, eeg_ch_names, start_time

    with Parallel(n_jobs=n_jobs, verbose=False) as pp:
        results = pp(delayed(_get_eeg_spectrogram)(df.iloc[i]) for i in tqdm(range(len(df))))

    output_folder = 'spectrograms'
    os.makedirs(output_folder, exist_ok=True)
    for i, res in enumerate(tqdm(results)):
        with open(os.path.join(output_folder, f'spec_{df.HashID.iloc[i]}.pickle'), 'wb') as ff:
            pickle.dump(res, ff)


if __name__=='__main__':
    main()

