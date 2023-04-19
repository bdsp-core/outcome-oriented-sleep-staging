from collections import defaultdict
from itertools import groupby
import datetime
import os
import numpy as np
import pandas as pd
from scipy.stats import skew
import mne
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size': 10})
import seaborn as sns
sns.set_style('ticks')


def plot(df_signals, params, sid):
    """
    """
    t0 = params['start_time']
    Fs = params['Fs']
    tt_h = df_signals.t_second.values/3600  # [hour]
    Tmax_sec = df_signals.t_second.max()  # [second]

    df_signals = df_signals.rename(columns={'chin1-chin2':'chin', 'airflow':'flow', 'e2-m2':'e2-m1', 'ekg':'ecg'})
    df_signals['sum'] = df_signals.abd + df_signals.chest
    if skew(df_signals.ecg.dropna()):
        df_signals['ecg'] = -df_signals.ecg
    print(df_signals.columns)

    spec_epoch_time = 4  # [s]
    spec_epoch_size = int(round(spec_epoch_time*Fs))
    spec_epoch_step = int(round(spec_epoch_time/2*Fs))
    eeg = df_signals['c4-m1'].values
    spec_epoch_start_ids = np.arange(0, len(eeg)-spec_epoch_size+1, spec_epoch_step)
    spec_epoch_start_times = spec_epoch_start_ids/Fs
    spec_epochs = np.array([eeg[x:x+spec_epoch_size] for x in spec_epoch_start_ids])
    NW = 3
    bw = NW*2/spec_epoch_time
    spec, spec_freq = mne.time_frequency.psd_array_multitaper(spec_epochs, Fs,
            fmin=0.5, fmax=15, bandwidth=bw, adaptive=False, low_bias=True,
            normalization='full', output='power', n_jobs=8, verbose=True)
    spec_db = 10*np.log10(spec)
    vmin = -10
    vmax = 15

    signal_types = ['eeg', 'eeg_spec', 'eog', 'emg', 'leg', 'ecg', 'snore', 'flow', 'ptaf']
    signal_types2 = ['spo2', 'hr']
    channels = { 'eeg': ['f4-m1', 'c4-m1', 'o2-m1'],
        'eog': ['e1-m2', 'e2-m1'],
        'emg': ['chin'],
        'leg': ['lat', 'rat'],
        'ecg': ['ecg'],
        'snore': ['snore'],
        'flow': ['flow', 'chest', 'sum', 'abd'],
        'ptaf':['ptaf'],
        'spo2':['spo2'],
        'hr':['hr'], }
    heights = {k:len(v) for k,v in channels.items()}
    heights['eeg_spec'] = 2
    colors = { 'eeg':'k', 'eog':'b', 'emg':'purple',
            'leg':(1/255,128/255,128/255),
            'ecg':'r', 'snore':(27/255,140/255,255/255),
            'flow':'k', 'ptaf':'k',
            'spo2':'k', 'hr':'k',
            'bg':(250/255,254/255,233/255),
            'epoch_border':(252/255,184/255,173/255),
            'stage_marker':(251/255,191/255,176/255),
            'position_marker':(188/255,188/255,188/255), }
    spaces = { 'eeg':100, 'eog':100,
            'emg':75, 'leg':100,
            'ecg':2000,
            'snore':20,
            'flow':np.nanpercentile(df_signals[channels['flow']].values.flatten(), 95)-np.nanpercentile(df_signals[channels['flow']].values.flatten(), 5),
            'ptaf':100, }
    notch_freq = 60
    freqs = {'eeg':[0.3,35], 'eog':[0.3,35],
            'emg':[10,100], 'leg':[10,100],
            'ecg':[0.3,100],
            'snore':[10,100],
            'flow':[0.1,15], }

    # filter
    for signal_type in signal_types:
        if signal_type in freqs:
            s = df_signals[channels[signal_type]].values.T
            s = mne.filter.notch_filter(s, Fs, notch_freq, verbose=False)
            lf, hf = freqs[signal_type]
            if hf>=Fs/2:
                hf = None
            s = mne.filter.filter_data(s, Fs, lf, hf, verbose=False)
            df_signals.loc[:, channels[signal_type]] = s.T

    epoch_time = 30
    durs = [epoch_time*1, epoch_time*3, epoch_time*10, epoch_time*20]
    figsize = (13,9.5)

    def _plot_signals(disp_start_time, dur, axs_=None):
        disp_end_time = disp_start_time+dur
        disp_start_idx = int(round(disp_start_time*Fs))
        disp_end_idx = int(round(disp_end_time*Fs))

        df_signals_disp = df_signals.iloc[disp_start_idx:disp_end_idx].reset_index(drop=True)
        epoch_borders = np.arange(0, df_signals.t_second.max(), epoch_time)
        ids = (epoch_borders>df_signals_disp.t_second.min())&(epoch_borders<df_signals_disp.t_second.max())
        epoch_borders = epoch_borders[ids]
        stage_marker_time = np.arange(epoch_time/2, df_signals.t_second.max(), epoch_time)
        ids = (stage_marker_time>df_signals_disp.t_second.min())&(stage_marker_time<df_signals_disp.t_second.max())
        stage_marker_time = stage_marker_time[ids]
        stage_marker = [params['stage_code'][x] for x in df_signals.stage.iloc[np.searchsorted(df_signals.t_second.values, stage_marker_time)]]
        position_marker = [params['body_position_code'][x] for x in df_signals.body_position.iloc[np.searchsorted(df_signals.t_second.values, stage_marker_time)]]
        epoch_ids = (stage_marker_time/epoch_time).astype(int)+1

        spec_disp_ids = (spec_epoch_start_times>=disp_start_time)&(spec_epoch_start_times+spec_epoch_time<=disp_start_time+dur)
        spec_db_disp = spec_db[spec_disp_ids]
        spec_epoch_start_times_disp = spec_epoch_start_times[spec_disp_ids]

        xticks = [disp_start_time, disp_end_time]
        xticklabels = [(t0+datetime.timedelta(seconds=int(x))).strftime('%H:%M:%S') for x in xticks]
        if dur == epoch_time:
            ticks = np.arange(np.ceil(df_signals_disp.t_second.min())+1, np.floor(df_signals_disp.t_second.max())+1)

        # signals
        gss1 = gs[0,0].subgridspec(len(signal_types)+len(signal_types2),1,hspace=0, height_ratios=[heights[x] for x in signal_types+signal_types2])
        if axs_ is None:
            axs = []
        for axi, signal_type in enumerate(signal_types):
            #print(signal_type)
            if axs_ is None:
                if axi==0:
                    ax = fig.add_subplot(gss1[axi,0]); ax0 = ax
                else:
                    ax = fig.add_subplot(gss1[axi,0], sharex=ax0)
                axs.append(ax)
            else:
                ax = axs_[axi]
                ax.clear()
            if signal_type=='eeg_spec':
                ax.imshow(spec_db_disp.T, origin='lower', aspect='auto',
                        cmap='turbo', vmin=vmin, vmax=vmax,
                        extent=(spec_epoch_start_times_disp.min(), spec_epoch_start_times_disp.max()+spec_epoch_time, spec_freq.min(), spec_freq.max()))
                yticks = [1, 4, 10, 13]
                for y in yticks:
                    ax.axhline(y, color='k', ls='--', lw=0.5)
                ax.set_yticks(yticks)
                ax.set_ylabel('Hz')
                #ax.yaxis.grid(True)
            else:
                channels_ = channels[signal_type]
                ch_space = spaces[signal_type]
                ch_color = colors[signal_type]
                Nch = len(channels_)
                ch_offsets = np.arange(Nch)*ch_space
                ylim = [-ch_space*0.5, (Nch-0.5)*ch_space]
                # epoch border
                for t in epoch_borders:
                    ax.axvline(t, color=colors['epoch_border'], lw=2, zorder=10)
                # stage marker
                if signal_type=='eeg':
                    for t,s,e in zip(stage_marker_time, stage_marker, epoch_ids):
                        if len(s)>0:
                            ax.text(t, np.mean(ylim), s, color=colors['stage_marker'], fontsize=50, zorder=11, ha='center', va='center')
                        ax.text(t-14.5, ylim[1]*1.1, e, color='b', zorder=11, ha='left', va='top')
                        ax.text(t+14.5, ylim[1]*1.1, e, color='b', zorder=11, ha='right', va='top')
                # position marker
                if signal_type=='flow':
                    for t,s in zip(stage_marker_time, position_marker):
                        if len(s)>0:
                            ax.text(t, np.mean(ylim), s, color=colors['position_marker'], fontsize=50, zorder=11, ha='center', va='center')
                ax.plot(df_signals_disp.t_second, df_signals_disp[channels_].values+ch_offsets[::-1], c=ch_color, clip_on=False, lw=0.5, zorder=20)
                # event annotation
                if signal_type in ['eeg', 'flow', 'leg']:
                    if signal_type=='eeg':
                        event = 'arousal'
                    elif signal_type=='flow':
                        event = 'resp'
                    elif signal_type=='leg':
                        event = 'limb'
                    cc = 0
                    for k,l in groupby(df_signals_disp[event]):
                        ll = len(list(l))
                        if k>0:
                            ax.plot(df_signals_disp.t_second[cc:cc+ll], df_signals_disp[channels_].values[cc:cc+ll]+ch_offsets[::-1], c=params[event+'_color'][k], clip_on=False, lw=0.5, zorder=21)
                            tb = ax.text(df_signals_disp.t_second[cc+ll//2], np.median(df_signals_disp[channels_].values[cc:cc+ll]+ch_offsets[::-1], axis=0).max()*1.05, params[event+'_code'][k], color=params[event+'_color'][k], ha='center', va='bottom', weight='bold', zorder=100)
                            tb.set_bbox(dict(facecolor='w', alpha=0.3, edgecolor='none'))
                        cc += ll
                # 1 second ticks if dur=30second
                if dur == epoch_time:
                    for x in ticks:
                        ax.axvline(x, c='gray', ls=':', lw=0.5, zorder=15)
                ax.set_ylim(ylim)
                ax.set_yticks(ch_offsets, labels=[ch.upper() for ch in channels_][::-1])

            ax.set_xlim(disp_start_time, disp_end_time)
            #if axi<len(signal_types)-1:
            ax.set_frame_on(False)
            plt.setp(ax.get_xticklabels(), visible=False)
            #else:
            #    ax.set_xticks(xticks)
            #    ax.set_xticklabels(xticklabels)
            #    sns.despine(left=True, right=True, top=True, bottom=False)
            #ax.set_facecolor(colors['bg'])

        # SpO2 and HR
        q1, q3 = np.nanpercentile(df_signals.spo2.values, (25,75))
        lb = q1-4*(q3-q1); ub = 100
        df_signals.loc[(df_signals.spo2<lb)|(df_signals.spo2>ub), 'spo2'] = np.nan
        all_hr = np.array(df_signals.hr.values)
        all_hr[np.isinf(all_hr)] = np.nan
        q1, q3 = np.nanpercentile(all_hr, (25,75))
        lb = q1-4*(q3-q1); ub = q3+4*(q3-q1)
        df_signals.loc[(df_signals.hr<lb)|(df_signals.hr>ub), 'hr'] = np.nan

        for axi, signal_type in enumerate(signal_types2):
            signal = df_signals_disp[signal_type]
            if np.all(np.isnan(signal)):
                ylim = [-2,2]
            else:
                ylim = [np.nanmin(signal)-2, np.nanmax(signal)+2]

            if axs_ is None:
                ax = fig.add_subplot(gss1[len(signal_types)+axi,0], sharex=ax0)
                axs.append(ax)
            else:
                ax = axs_[len(signal_types)+axi]
                ax.clear()
            for t in epoch_borders: # epoch border
                ax.axvline(t, color=colors['epoch_border'], lw=2, zorder=10)
            ax.step(df_signals_disp.t_second, signal, color=colors[signal_type], lw=1, zorder=11, where='post')
            signal_ = np.array(signal); signal_[np.isnan(signal_)] = -1
            marker_ids = []
            cc = 0
            for k,l in groupby(signal_):
                ll = len(list(l))
                if k>0:
                    marker_ids.append(cc+ll//2)
                cc += ll
            # only show local max or min points
            marker_ids = [x for ii,x in enumerate(marker_ids) if ii==0 or ii==len(marker_ids)-1 or signal_[marker_ids[ii-1]]>signal_[x] and signal_[x]<signal_[marker_ids[ii+1]] or signal_[marker_ids[ii-1]]<signal_[x] and signal_[x]>signal_[marker_ids[ii+1]]]
            ax.scatter(df_signals_disp.t_second.iloc[marker_ids], signal[marker_ids], c=colors[signal_type], s=15, zorder=12)
            for idx in marker_ids:
                ax.text(df_signals_disp.t_second.iloc[idx], signal[idx]+1, str(int(round(signal[idx]))), c='b', weight='bold', va='bottom', ha='center')
            # 1 second ticks if dur=30second
            if dur == epoch_time:
                for x in ticks:
                    ax.axvline(x, c='gray', ls=':', lw=0.5, zorder=15)
            ax.set_ylim(ylim)
            ax.set_yticks([])
            ax.set_ylabel(signal_type.upper(), rotation=0, va='center', ha='right')
            #ax.set_xlim(disp_start_time, disp_end_time)
            if axi<len(signal_types2)-1:
                ax.set_frame_on(False)
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels, color='b')
                sns.despine(left=True, right=True, top=True, bottom=False)
        if axs_ is None:
            return axs

    # bottom overview
    def _plot_overview(disp_start_time, dur):
        disp_end_time = disp_start_time+dur

        xticks = np.arange(int(tt_h.max())+1)
        if xticks[-1]<tt_h.max():
            xticks = np.r_[xticks, tt_h.max()]
        xticklabels = [(t0+datetime.timedelta(seconds=int(x*3600))).strftime('%H:%M') for x in xticks]

        bar_names = ['stage', 'body_position', 'resp', 'arousal', 'limb']#, 'SpO2', 'cpres']
        bar_names_disp = ['STG', 'BODY', 'RESP', 'AROU', 'LIMB']#, 'SpO2', 'PRES']
        gss2 = gs[1,0].subgridspec(len(bar_names)+2,1,hspace=0)
        start_locators = []
        end_locators = []
        for axi, bn in enumerate(bar_names):
            #print(bn)
            if axi==0:
                ax = fig.add_subplot(gss2[axi,0]); ax_bottom0 = ax
            else:
                ax = fig.add_subplot(gss2[axi,0], sharex=ax_bottom0)
            bar_var = df_signals[bn].values
            bar_var[np.isinf(bar_var)] = -1
            bar_var[np.isnan(bar_var)] = -1
            bar_code  = params[bn+'_code']
            bar_color = params[bn+'_color']
            cc = 0
            for k,l in groupby(bar_var):
                ll = len(list(l))
                if k>0:
                    ax.add_patch(Rectangle((tt_h[cc],0), tt_h[cc+ll-1]-tt_h[cc], 1, color=bar_color[k]))
                    if ll>=Fs*epoch_time*5 and len(bar_code[k])>0:
                        ax.text((tt_h[cc]+tt_h[cc+ll-1])/2, 0.5, bar_code[k], ha='center', va='center', color='k', fontsize=6)
                cc += ll
            # plot time locator
            a = ax.axvline(disp_start_time/3600, c='k', ls=':', lw=1, zorder=1000)
            b = ax.axvline(disp_end_time/3600, c='k', ls=':', lw=1, zorder=1000)
            start_locators.append(a)
            end_locators.append(b)
            ax.set_ylim(-0.1,1.1)
            ax.set_ylabel(bar_names_disp[axi], rotation=0, va='center', ha='right')
            ax.set_yticks([])
            ax.set_xlim(tt_h.min(), tt_h.max())
            #if axi<len(bar_names)-1:
            ax.set_frame_on(False)
            #plt.setp(ax.get_xticks(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            #else:

        # overall SpO2
        ax = fig.add_subplot(gss2[len(bar_names),0], sharex=ax_bottom0)
        ax.plot(tt_h, df_signals.spo2, c='k', lw=0.5)
        # plot time locator
        a = ax.axvline(disp_start_time/3600, c='k', ls=':', lw=1, zorder=1000)
        b = ax.axvline(disp_end_time/3600, c='k', ls=':', lw=1, zorder=1000)
        start_locators.append(a)
        end_locators.append(b)
        ax.set_ylim(df_signals.spo2.min()-2, 100)
        ax.set_ylabel('SpO2', rotation=0, va='center', ha='right')
        ax.set_yticks([])
        ax.set_xlim(tt_h.min(), tt_h.max())
        ax.set_frame_on(False)
        plt.setp(ax.get_xticklabels(), visible=False)

        # overall HR
        ax = fig.add_subplot(gss2[len(bar_names)+1,0], sharex=ax_bottom0)
        ax.plot(tt_h, df_signals.hr, c='k', lw=0.5)
        # plot time locator
        a = ax.axvline(disp_start_time/3600, c='k', ls=':', lw=1, zorder=1000)
        b = ax.axvline(disp_end_time/3600, c='k', ls=':', lw=1, zorder=1000)
        start_locators.append(a)
        end_locators.append(b)
        ax.set_ylim(df_signals.hr.min()-2, df_signals.hr.max()+2)
        ax.set_ylabel('HR', rotation=0, va='center', ha='right')
        ax.set_yticks([])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        sns.despine(left=True, right=True, top=True, bottom=False)

        return start_locators, end_locators

    class MyPlot:
        def __init__(self):#, df_signals, axs, locators):
            self.disp_start_time = 0
            self.dur_idx = 0
            self.disp_start_times = np.arange(0, Tmax_sec, epoch_time)
            self.current_annotating_event = ''
            self.key2event = {'i':'spindle',
                              'd':'slow wave',
                              'y':'rapid eye movement'}
            self.annot_events = defaultdict(list)

        def on_close(self, event):
            time = datetime.datetime.now().strftime('%H-%M-%S %m-%d-%Y')
            for k,v in self.annot_events.items():
                annot_folder = 'manual_annotations'
                os.makedirs(annot_folder, exist_ok=True)
                path = os.path.join(annot_folder, f'annot_{k}_{time}_{sid}.csv')
                print(f'saving {k} annotations to {path}...')
                v2 = np.array([x for x in v if len(x)==2 and x[0]<x[1]])
                if len(v2)>0:
                    df_ = pd.DataFrame(data=v2, columns=['Start','End'])
                    df_.to_csv(path, index=False)
            print(f'close at {time} for {sid}')

        def on_click(self, event):
            if self.current_annotating_event=='':
                return

            if event.button==1:
                if len(self.annot_events[self.current_annotating_event])>0 and len(self.annot_events[self.current_annotating_event][-1])==1:
                    self.annot_events[self.current_annotating_event][-1][0] = event.xdata
                else:
                    self.annot_events[self.current_annotating_event].append([event.xdata])
                print(self.annot_events)
            elif event.button==3:
                if len(self.annot_events[self.current_annotating_event])>0:
                    if len(self.annot_events[self.current_annotating_event][-1])==1:
                        self.annot_events[self.current_annotating_event][-1].append(event.xdata)
                    else:
                        self.annot_events[self.current_annotating_event][-1][1] = event.xdata
                    print(self.annot_events)

        def on_press(self, event):
            changed = False
            if event.key == 'left' and self.disp_start_time-epoch_time>=0:
                self.disp_start_time -= epoch_time
                print('go to the previous epoch')
                changed = True
            elif event.key == 'right' and self.disp_start_time+epoch_time*2<=Tmax_sec:
                self.disp_start_time += epoch_time
                print('go to the next epoch')
                changed = True
            elif event.key == 'pageup' and self.disp_start_time-durs[self.dur_idx]*10>=0:
                self.disp_start_time -= durs[self.dur_idx]*10
                print('go to 10x previous epochs')
                changed = True
            elif event.key == 'pagedown' and self.disp_start_time+durs[self.dur_idx]*11<=Tmax_sec:
                self.disp_start_time += durs[self.dur_idx]*10
                print('go to 10x next epochs')
                changed = True
            elif event.key == 'home':
                self.disp_start_time = 0
                print('go to the first epochs')
                changed = True
            elif event.key == 'end':
                self.disp_start_time = self.disp_start_times[self.disp_start_times+durs[self.dur_idx]<=Tmax_sec][-1]
                print('go to the last epochs')
                changed = True
            elif event.key == 'down' and self.dur_idx-1>=0:
                self.dur_idx -= 1
                print('make duration shorter')
                changed = True
            elif event.key == 'up' and self.dur_idx+1<len(durs) and self.disp_start_time+durs[self.dur_idx+1]<=Tmax_sec:
                self.dur_idx += 1
                print('make duration longer')
                changed = True
            elif event.key in ['1', '2', '3', 'r', 'w', 'a', 'p', 'l', 'b']:
                if event.key in ['1', '2', '3', 'r', 'w']:
                    idx = ['1', '2', '3', 'r', 'w'].index(event.key)
                    stage = ['1','2','3','R','W'][idx]
                    event_mask = df_signals.stage==params['stage_txt2code'][stage]
                elif event.key =='a':
                    event_mask = df_signals.arousal>0
                elif event.key =='p':
                    event_mask = df_signals.resp>0
                elif event.key =='l':
                    event_mask = df_signals.limb>0
                elif event.key =='b':
                    event_mask = df_signals.body_position>0
                event_ids = np.where(event_mask&(df_signals.t_second>self.disp_start_time+durs[self.dur_idx]/2))[0]
                if len(event_ids)>0:
                    event_time = df_signals.t_second.iloc[event_ids[0]]
                    disp_start_time = self.disp_start_times[self.disp_start_times<=event_time].max()
                    if disp_start_time+durs[self.dur_idx]<=Tmax_sec:
                        self.disp_start_time = disp_start_time
                        print('go to the closest next event')
                        changed = True
            elif event.key in ['!', '@', '#', 'R', 'W', 'A', 'P', 'L', 'B']:
                if event.key in ['!', '@', '#', 'R', 'W']:
                    idx = ['!', '@', '#', 'R', 'W'].index(event.key)
                    stage = ['1','2','3','R','W'][idx]
                    event_mask = df_signals.stage==params['stage_txt2code'][stage]
                elif event.key =='A':
                    event_mask = df_signals.arousal>0
                elif event.key =='P':
                    event_mask = df_signals.resp>0
                elif event.key =='L':
                    event_mask = df_signals.limb>0
                elif event.key =='B':
                    event_mask = df_signals.body_position>0
                event_ids = np.where(event_mask&(df_signals.t_second<self.disp_start_time))[0]
                if len(event_ids)>0:
                    event_time = df_signals.t_second.iloc[event_ids[-1]]
                    self.disp_start_time = self.disp_start_times[self.disp_start_times<=event_time].max()
                    print('go to the closest previous event')
                    changed = True
            # for annotating events
            elif event.key in ['i', 'd', 'y']:
                self.current_annotating_event = self.key2event[event.key]
                print(f'Annotating {self.current_annotating_event} starts')
            elif event.key in ['I', 'D', 'Y']:
                self.current_annotating_event = 'non-'+self.key2event[event.key.lower()]
                print(f'Annotating non-{self.current_annotating_event} starts')
            elif event.key=='x':
                self.current_annotating_event = ''
                print(f'Annotation stops')
            # for annotating non-events
            if changed:
                #plt.clf()
                for ax in axs:
                    ax.clear()
                _plot_signals(self.disp_start_time, durs[self.dur_idx], axs_=axs)
                #self.ax.set_xlim(self.disp_start_time, self.disp_start_time+durs[self.dur_idx])
                for loc in locators[0]:
                    loc.set_data(([self.disp_start_time/3600]*2, [0,1]))
                for loc in locators[1]:
                    loc.set_data(([(self.disp_start_time+durs[self.dur_idx])/3600]*2, [0,1]))
                fig.canvas.draw()

    plt.close()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2,1,height_ratios=(3.5,1))

    axs = _plot_signals(0, durs[0])
    locators = _plot_overview(0, durs[0])

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.975, top=0.98, bottom=0.04, hspace=0.09)

    myplot = MyPlot()
    fig.canvas.mpl_connect('key_press_event', myplot.on_press)
    fig.canvas.mpl_connect('button_press_event', myplot.on_click)
    fig.canvas.mpl_connect('close_event', myplot.on_close)

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


if __name__=='__main__':
    import re
    from load_data import load_data

    #import sys
    #subject_folder = sys.argv[1]
    folders = [x for x in os.listdir('.') if re.match('[0-9a-z]{64}_[0-9]{8}_[0-9]{9}',x)]
    for subject_folder in tqdm(folders):
        df_signals, params = load_data(subject_folder)
        plot(df_signals, params, subject_folder)
        #TODO, show age and sex

