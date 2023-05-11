import datetime
import os
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
import seaborn as sns
sns.set_style('ticks')
import sys
sys.path.insert(0, '/data/interesting_side_projects/sleep_general')
from mgh_sleeplab import load_mgh_signal, annotations_preprocess, vectorize_sleep_stages
sys.path.insert(0, '..')
from pattern_detection import get_spindle_peak_freq, my_spindle_detect, my_sw_detect, my_rem_detect
        

class MyPlot:
    def __init__(self, sid, event_name, df_signals, df_detect, Fs, start_time, sleep_stages, freq_sw, freq_rem, save_folder='manual_check'):
        self.sid = sid
        self.event_name = event_name
        self.df_signals = df_signals
        self.df_detect = df_detect
        self.Fs = Fs
        self.start_time = start_time
        self.sleep_stages = sleep_stages
        self.freq_sw = freq_sw
        self.freq_rem = freq_rem
        self.save_folder = save_folder
        self.rowi = 0
        self.is_event = []

    def on_press(self, event):
        if event.key in ['0', '1'] and len(self.is_event)<len(self.df_detect):
            self.is_event.append(int(event.key))
            if self.rowi<len(self.df_detect):
                self.plot(redraw=True)
                self.fig.canvas.draw()

            print(self.is_event)
            df = self.df_detect.copy()
            df = df[:len(self.is_event)]
            df['ManualOK'] = self.is_event
            path = os.path.join(self.save_folder, f'manual_check_{self.event_name}_{self.sid}.csv')
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
        figsize = (11.4,6)

        start_time  = self.df_detect.Start.iloc[self.rowi]
        end_time    = self.df_detect.End.iloc[self.rowi]
        event_ch    = self.df_detect.Channel.iloc[self.rowi]

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
        sleep_stage = sleep_stage[pd.notna(sleep_stage)]
        sleep_stage_txt = ','.join([stage_num2txt[x] for x in sleep_stage])

        if self.event_name=='spindle':
            peak_freq   = self.df_detect.Frequency.iloc[self.rowi]
            signal_f_disp = mne.filter.filter_data(signal_disp, self.Fs, peak_freq-1, peak_freq+1, verbose=False)
        elif self.event_name=='slow_wave':
            signal_f_disp = mne.filter.filter_data(signal_disp, self.Fs, self.freq_sw[0], self.freq_sw[1], verbose=False)
        elif self.event_name=='rem':
            signal_f_disp = mne.filter.filter_data(signal_disp, self.Fs, self.freq_rem[0], self.freq_rem[1], verbose=False)
            
        spec_epoch_start_ids = np.arange(0, signal_disp.shape[1]-spec_epoch_size+1, spec_epoch_step)
        spec_epoch_start_times = spec_epoch_start_ids/self.Fs
        spec_epochs = np.array([signal_disp[:,x:x+spec_epoch_size] for x in spec_epoch_start_ids])
        spec, spec_freq = mne.time_frequency.psd_array_multitaper(spec_epochs, self.Fs,
                fmin=0.5, fmax=16, bandwidth=bw, adaptive=False, low_bias=True,
                normalization='full', n_jobs=n_jobs, verbose=False)
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
        yticks = [1,4,8,10,11,13.5,16]
        yticklabels = ['1','4','8','10','11','13.5','16']
        ax.set_yticks(yticks, labels=yticklabels)
        for y in yticks:
            ax.axhline(y, color='k', ls='--', lw=1)

        self.rowi += 1
        if not redraw:
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.04)
            self.fig.canvas.mpl_connect('key_press_event', self.on_press)
            plt.show()


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


def main(pattern):
    import re
    
    random_seed = 2023
    folders = [x for x in os.listdir('.') if re.match('[0-9a-z]{64}_[0-9]{8}_[0-9]{9}',x)]
    df_annot = pd.read_csv('annotations_sleep_stages.zip', compression='zip')
    df_annot['DOVshifted'] = pd.to_datetime(df_annot.DOVshifted)

    freq_sw = [0.5,2]
    freq_rem = [0.5,5]
    
    eeg_ch_names = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1']#, 'o1-m2', 'o2-m1']
    eog_ch_names = ['e1-m2', 'e2-m1']
    ch_names = eeg_ch_names + eog_ch_names
    eeg_ch_names_re = ['f3-', 'f4-', 'c3-', 'c4-']#, 'o1-', 'o2-']
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
        signals = pd.DataFrame(data=np.vstack([eeg,eog]).T, columns=ch_names)
        
        if pattern=='spindle':
            epoch_size = int(round(30*Fs))
            start_ids = np.arange(0, len(signals)-epoch_size+1, epoch_size)
            sleep_stages_epochs = sleep_stages[start_ids]
            epochs = np.array([eeg[:,x:x+epoch_size] for x in start_ids])
            spec, freq = mne.time_frequency.psd_array_multitaper(epochs, Fs, fmin=0.3, fmax=35, bandwidth=0.5, adaptive=False, low_bias=True, normalization='full', verbose=False)
            spindle_peak_freq = get_spindle_peak_freq(spec, freq, sleep_stages_epochs)
            print(spindle_peak_freq)
            spindle_peak_freq[np.isnan(spindle_peak_freq)] = np.nanmedian(spindle_peak_freq)
            
        if pattern=='spindle':
            res = my_spindle_detect(eeg, sleep_stages, Fs, eeg_ch_names, include=[2],
                freq_sp=[[spindle_peak_freq[chi]-1,spindle_peak_freq[chi]+1] for chi in range(len(eeg))],
                thresh={'corr':0.6, 'rel_pow':0.1, 'rms':1.5, 'amp':np.inf}, verbose=False)
            if len(res)==0:
                continue
            np.random.seed(random_seed+si*1000)
            Nrand = len(res)//50 if len(res)>=500 else min(20,len(res))
            res = res.iloc[np.sort(np.random.choice(len(res), Nrand, replace=False))]
                
        elif pattern=='slow_wave':
            res = my_sw_detect(eeg, sleep_stages, Fs, eeg_ch_names, include=[1,2],
                freq_sw=freq_sw, amp_neg=[40,200],  amp_pos=[10,150], amp_ptp=[70, 350],
                verbose=False)
            if len(res)==0:
                continue
            np.random.seed(random_seed+si*1000)
            Nrand = len(res)//50 if len(res)>=500 else min(20,len(res))
            res = res.iloc[np.sort(np.random.choice(len(res), Nrand, replace=False))].reset_index(drop=True)

        elif pattern=='rem':
            res = my_rem_detect(eog[0], eog[1], sleep_stages, Fs, eog_ch_names[0], include=[1,2,3,4], amplitude=[50,325], duration=[0.3,1.2], freq_rem=freq_rem, verbose=False)
            if len(res)==0:
                continue
            res['IdxChannel'] = res.IdxChannel+len(eeg)
            np.random.seed(random_seed+si*1000)
            Nrand = len(res)//50 if len(res)>=500 else min(20,len(res))
            res = res.iloc[np.sort(np.random.choice(len(res), Nrand, replace=False))].reset_index(drop=True)
            
        print(len(res))

        if len(res)>0:
            myplot = MyPlot(subject_folder, pattern, signals, res, Fs, start_time, sleep_stages, freq_sw, freq_rem)
            myplot.plot()

    save_folder = myplot.save_folder
    df_res = []
    for subject_folder in folders:
        path = os.path.join(save_folder, f'manual_check_{pattern}_{subject_folder}.csv')
        df = pd.read_csv(path)
        df.insert(0, 'SID', subject_folder)
        df_res.append(df)
    df_res = pd.concat(df_res, axis=0, ignore_index=True)
    df_res.to_excel(os.path.join(save_folder, f'manual_check_{pattern}.xlsx'), index=False)


if __name__=='__main__':
    pattern = sys.argv[1].lower().strip()
    assert pattern in ['spindle', 'slow_wave', 'rem']
    main(pattern)

