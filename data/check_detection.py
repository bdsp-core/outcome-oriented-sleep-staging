import datetime
import os
import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, argrelmax
from scipy.stats import linregress
import mne
from tqdm import tqdm
import yasa
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
import seaborn as sns
sns.set_style('ticks')
import sys
sys.path.insert(0, '../../sleep_general')
from mgh_sleeplab import load_mgh_signal, annotations_preprocess, vectorize_sleep_stages


class MyPlot:
    def __init__(self, sid, event_name, df_signals, df_detect, Fs, start_time, sleep_stages, save_folder='manual_check'):
        self.sid = sid
        self.event_name = event_name
        self.df_signals = df_signals
        self.df_detect = df_detect
        self.Fs = Fs
        self.start_time = start_time
        self.sleep_stages = sleep_stages
        self.save_folder = save_folder
        self.rowi = 0
        self.is_event = []

    def on_press(self, event):
        changed = False
        if event.key in ['0', '1'] and len(self.is_event)<len(self.df_detect):
            self.is_event.append(int(event.key))
            changed = True
        if changed:
            self.rowi += 1
            self.plot(redraw=True)
            self.fig.canvas.draw()

            print(self.is_event)
            df = self.df_detect.copy()
            df = df[:len(self.is_event)]
            df['ManualOK'] = self.is_event
            path = os.path.join(self.save_folder, f'manual_check_{self.sid}.csv')
            os.makedirs(self.save_folder, exist_ok=True)
            df.to_csv(path, index=False)

    def plot(self, redraw=False):
        print(f'{self.rowi+1}/{len(self.df_detect)}')
        stage_num2txt = {5:'W', 4:'R', 3:'N1', 2:'N2', 1:'N3'}
        tt = np.arange(len(self.df_signals))/self.Fs
        disp_time = 20  # [second]
        spec_epoch_time = 4  # [s]
        NW = 3
        spec_epoch_size = int(round(spec_epoch_time*self.Fs))
        spec_epoch_step = int(round(spec_epoch_time/2*self.Fs))
        bw = NW*2/spec_epoch_time
        n_jobs = 8
        vmin = -10
        vmax = 15
        figsize = (11.4,5.6)

        start_time  = self.df_detect.Start.iloc[self.rowi]
        end_time    = self.df_detect.End.iloc[self.rowi]
        event_ch    = self.df_detect.Channel.iloc[self.rowi]
        peak_freq   = self.df_detect.Frequency.iloc[self.rowi]

        middle_time = (start_time+end_time)/2
        start_time2 = max(0, middle_time-disp_time/2)
        end_time2   = min(len(self.df_signals)/self.Fs, middle_time+disp_time/2)
        start_idx   = int(round(start_time2*self.Fs))
        end_idx     = int(round(end_time2*self.Fs))
        signal_disp = self.df_signals.iloc[start_idx:end_idx].values.T
        tt_disp     = tt[start_idx:end_idx]
        event_start_id = np.searchsorted(tt_disp, start_time)
        event_end_id   = np.searchsorted(tt_disp, end_time)+1
        event_ch_id = list(self.df_signals.columns).index(event_ch)
        sleep_stage = np.unique(self.sleep_stages[start_idx:end_idx])
        sleep_stage_txt = ','.join([stage_num2txt[x] for x in sleep_stage])

        if self.event_name=='spindle':
            signal_f_disp = mne.filter.filter_data(signal_disp, self.Fs, peak_freq-1, peak_freq+1, verbose=False)

        spec_epoch_start_ids = np.arange(0, signal_disp.shape[1]-spec_epoch_size+1, spec_epoch_step)
        spec_epoch_start_times = spec_epoch_start_ids/self.Fs
        spec_epochs = np.array([signal_disp[:,x:x+spec_epoch_size] for x in spec_epoch_start_ids])
        spec, spec_freq = mne.time_frequency.psd_array_multitaper(spec_epochs, self.Fs,
                fmin=0.5, fmax=16, bandwidth=bw, adaptive=False, low_bias=True,
                normalization='full', output='power', n_jobs=n_jobs, verbose=False)
        spec_db = 10*np.log10(spec)

        xticks = np.arange(0, disp_time+1)+start_time2
        xticklabels = [(self.start_time+datetime.timedelta(seconds=x)).strftime('%H:%M:%S') if xi%10==0 else '' for xi,x in enumerate(xticks)]

        if not redraw:
            plt.close()
            self.fig = plt.figure(figsize=figsize)
            gs = self.fig.add_gridspec(3,1,height_ratios=(6,1,3))
            gss = gs[0,0].subgridspec(self.df_signals.shape[1],1,hspace=0)
            self.ax_signal = []

        for chi in range(signal_disp.shape[0]):
            if redraw:
                ax = self.ax_signal[chi]
                ax.clear()
            else:
                if chi==0:
                    ax = self.fig.add_subplot(gss[chi,0])
                    ax0 = ax
                else:
                    ax = self.fig.add_subplot(gss[chi,0], sharex=ax0, sharey=ax0)
            ax.plot(tt_disp, signal_disp[chi], c='k', lw=0.5)
            if chi==event_ch_id:
                ax.plot(tt_disp[event_start_id:event_end_id], signal_disp[chi][event_start_id:event_end_id], c='r', lw=0.5)
            ax.set_xticks(xticks, labels=xticklabels)
            ax.xaxis.grid(True)
            #ax.set_xlabel('time (s)')
            ax.set_ylabel(self.df_signals.columns[chi].upper(), rotation=0)
            sns.despine()
            #if chi!=signal_disp.shape[0]-1:
            plt.setp(ax.get_xticklabels(), visible=False)
            if not redraw:
                self.ax_signal.append(ax)

        if redraw:
            ax = self.ax_signal[signal_disp.shape[0]]
            ax.clear()
        else:
            ax = self.fig.add_subplot(gs[1,0], sharex=ax0)
            self.ax_signal.append(ax)
        ax.plot(tt_disp, signal_f_disp[event_ch_id], c='k', lw=0.5)
        ax.plot(tt_disp[event_start_id:event_end_id], signal_f_disp[event_ch_id][event_start_id:event_end_id], c='r', lw=0.5)
        ax.set_xticks(xticks, labels=xticklabels)
        ax.xaxis.grid(True)
        ax.set_ylabel(event_ch.upper(), rotation=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        sns.despine()

        if redraw:
            ax = self.ax_signal[signal_disp.shape[0]+1]
            ax.clear()
        else:
            ax = self.fig.add_subplot(gs[2,0], sharex=ax0)
            self.ax_signal.append(ax)
        ax.imshow(spec_db[:,event_ch_id].T, aspect='auto', origin='lower', cmap='turbo', vmin=vmin, vmax=vmax,
                extent=(tt_disp.min(), tt_disp.max(), spec_freq.min(), spec_freq.max()), interpolation='bilinear')
        ax.axvline(tt_disp[event_start_id], c='r', lw=1)
        ax.axvline(tt_disp[event_end_id], c='r', lw=1)
        ax.set_xticks(xticks, labels=xticklabels)
        #ax.xaxis.grid(True)
        yticks = [1,4,8,10,11,peak_freq,16]
        yticklabels = ['1','4','8','10','11',f'{peak_freq:.1f}','16']
        ax.set_yticks(yticks, labels=yticklabels)
        for y in yticks:
            ax.axhline(y, color='k', ls='--', lw=1)

        if not redraw:
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.04)
            self.fig.canvas.mpl_connect('key_press_event', self.on_press)
            plt.show()


def load_data(subject_folder, base_folder, ch_names, df_annot=None):
    # load signal
    signal_path = os.path.join(base_folder, subject_folder, 'Shifted_signal_'+subject_folder+'.mat')
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


def myapplyall(x, m):
    from itertools import groupby
    x2 = np.array(x)
    x2[np.isnan(x2)] = -1
    x3 = np.zeros_like(x)
    cc = 0
    for k,l in groupby(x2):
        ll = len(list(l))
        x3[cc:cc+ll] = m.get(k,-1)
        cc += ll
    return x3


def get_spindle_peak_freq(eeg, sleep_stages, Fs):
    freq_range = [11,16]
    epoch_time = 30  # [second]
    epoch_size = int(round(epoch_time*Fs))
    epoch_step = int(round(epoch_time*Fs))
    start_ids = np.arange(0, eeg.shape[1]-epoch_size+1, epoch_step)
    N2_ids = np.where(sleep_stages[start_ids]==2)[0]
    if len(N2_ids)<10:
        #TODO based on age norm
        raise ValueError
    epochs = np.array([eeg[:,x:x+epoch_size] for x in start_ids])
    NW = 10
    bw = 2*NW/epoch_time
    n_jobs = 8
    spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs,
            fmin=10, fmax=20, bandwidth=bw, adaptive=False, low_bias=True,
            normalization='full', output='power', n_jobs=n_jobs, verbose=False)
    spec_db = 10*np.log10(spec)
    spec_db_N2 = spec_db[N2_ids].mean(axis=0)

    peak_freqs = []
    for chi in range(spec_db_N2.shape[0]):
        slope, intercept, r, p, se = linregress(freq, spec_db_N2[chi])
        spec_ = spec_db_N2[chi]-freq*slope-intercept
        spec_ = savgol_filter(spec_, 45, 3, mode='nearest')
        peak_id = argrelmax(spec_)[0]
        if len(peak_id)>=1:
            peak_id = peak_id[np.argmax(spec_[peak_id])]
            peak_freq = freq[peak_id]
            ids = np.where((freq>=peak_freq-0.5)&(freq<=peak_freq+0.5))[0]
            spec_ = savgol_filter(spec_db_N2[chi], 45, 3, mode='nearest')
            peak_id2 = ids[argrelmax(spec_[ids])[0]]
            if len(peak_id2)>=1:
                peak_id2 = peak_id2[np.argmax(spec_[peak_id2])]
                peak_freq = freq[peak_id2]
        else:
            peak_freq = np.nan
        if peak_freq<freq_range[0] or peak_freq>freq_range[1]:
            peak_freq = np.nan
        peak_freqs.append(peak_freq)
    return np.array(peak_freqs)


def main(pattern):
    random_seed = 2023
    folders = [x for x in os.listdir('.') if re.match('[0-9a-z]{64}_[0-9]{8}_[0-9]{9}',x)]
    df_annot = pd.read_csv('annotations_sleep_stages.zip', compression='zip')
    df_annot['DOVshifted'] = pd.to_datetime(df_annot.DOVshifted)

    """
-2 = Unscored
-1 = Artefact / Movement
0 = Wake
1 = N1 sleep
2 = N2 sleep
3 = N3 sleep
4 = REM sleep
    """
    yasa_stage_mapping = {5:0,4:4,3:1,2:2,1:3,-1:-1}
    eeg_ch_names = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1']
    eog_ch_names = ['e1-m2', 'e2-m1']
    ch_names = eeg_ch_names + eog_ch_names
    eeg_ch_names_re = ['f3-', 'f4-', 'c3-', 'c4-', 'o1-', 'o2-']
    eog_ch_names_re = ['e1-', 'e2-']
    ch_names_re = eeg_ch_names_re + eog_ch_names_re

    for si, subject_folder in enumerate(tqdm(folders)):
        signals, sleep_stages, params = load_data(subject_folder, '.', ch_names_re, df_annot=df_annot)
        Fs = params['Fs']
        start_time = params['start_time']
        signals = signals.rename(columns={k:v for k,v in zip(params['channel_names'],ch_names)})

        eeg = signals[eeg_ch_names].values.T
        eog = signals[eog_ch_names].values.T
        eeg = filter_signal(eeg, Fs, 60, [0.3, 35])
        eog = filter_signal(eog, Fs, 60, [0.3, 35])

        # convert to yasa stage encoding
        sleep_stages2 = myapplyall(sleep_stages, yasa_stage_mapping)

        if pattern=='spindle':
            spindle_peak_freq = get_spindle_peak_freq(eeg, sleep_stages, Fs)
            print(spindle_peak_freq)
            spindle_peak_freq[np.isnan(spindle_peak_freq)] = np.nanmedian(spindle_peak_freq)
            res = []
            for chi in range(len(eeg)):
                res_ = yasa.spindles_detect( eeg[[chi]], sf=Fs, ch_names=[eeg_ch_names[chi]],
                    hypno=sleep_stages2, include=[2], freq_sp=[spindle_peak_freq[chi]-1,spindle_peak_freq[chi]+1], freq_broad=[1,30],
                    duration=[0.5,2], min_distance=500,
                    thresh={'corr':0.6, 'rel_pow':0.1, 'rms':1.5},
                    multi_only=False, remove_outliers=False, verbose=False)
                res_ = res_.summary()
                np.random.seed(random_seed+si*1000+chi)
                res_ = res_.iloc[np.sort(np.random.choice(len(res_), len(res_)//50, replace=False))]
                res.append( res_ )
            res = pd.concat(res, axis=0, ignore_index=True)
            print(res.shape)

            sig = pd.DataFrame(data=eeg.T, columns=eeg_ch_names)
        myplot = MyPlot(subject_folder, pattern, sig, res, Fs, start_time, sleep_stages)
        myplot.plot()


if __name__=='__main__':
    pattern = sys.argv[1].lower().strip()
    assert pattern in ['spindle', 'slow_wave', 'rem']
    main(pattern)

