import datetime
from itertools import groupby
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


def main():
    epoch_times = [15]#,10,5]
    random_state = 2023
    aasm_stages = ['N3', 'N2', 'N1', 'R', 'W']
    ch_names_combined = ['frontal', 'central', 'occipital']
    chi2chid = [[0,1],[2,3],[4,5]]
    new_state_colors = matplotlib.colormaps['tab10']
    aasm_stage_colors = {
            'N3':(3/255,203/255,255/255),
            'N2':(2/255,128/255,2/255),
            'N1':(255/255,255/255,0/255),
            'R':(255/255,102/255,155/255),
            'W':(0/255,0/255,0/255)}

    df = pd.read_csv('../data/mastersheet_matched_Dementia_alloutcomes.csv')

    for epoch_time in epoch_times:
        print(f'epoch_time = {epoch_time}s')
        figure_folder = f'figures_epochtime{epoch_time}s/signals'
        os.makedirs(figure_folder, exist_ok=True)

        df_feat_all = pd.read_csv(f'../data/features_epoch{epoch_time}s.csv.zip')
        sleep_stages_all = df_feat_all.SleepStage.values
        feat_cols = [
            'alpha_rel_bp_F', 'alpha_rel_bp_C', 'alpha_rel_bp_O',
            'theta_rel_bp_F', 'theta_rel_bp_C', 'theta_rel_bp_O',
            'has_spindle_F', 'has_spindle_C', 'has_spindle_O',
            'sw_amp_ptp_F', 'sw_amp_ptp_C', 'sw_amp_ptp_O',
            'sw_perc_F', 'sw_perc_C', 'sw_perc_O',
            'has_rem', 'emg_env_rank_mean', ]
        feat_cols2 = [
            r'$\alpha$ bp F', r'$\alpha$ bp C', r'$\alpha$ bp O',
            r'$\theta$ bp F', r'$\theta$ bp C', r'$\theta$ bp O',
            'spindle F', 'spindle C', 'spindle O',
            'sw amp F', 'sw amp C', 'sw amp O',
            'sw % F', 'sw % C', 'sw % O',
            'REM', 'EMG env', ]
        feat_lb, feat_ub = np.nanpercentile(df_feat_all[feat_cols], (1,99), axis=0)
        df_feat_all.loc[:,feat_cols] = np.clip((df_feat_all[feat_cols].values-feat_lb)/(feat_ub-feat_lb), 0, 1)

        with open(os.path.join(f'results_new_multiple_outcomes_epoch{epoch_time}s', 'results.pickle'), 'rb') as ff:
            res = pickle.load(ff)
        sids = res['sids_te']
        assert np.all(pd.unique(sids)==df.HashID)
        y = res['yte']
        yp = res['yp_final_train']

        Zp_all = np.concatenate(res['Zp'], axis=0)
        n_state = Zp_all.shape[1]
        Z_all = np.argmax(Zp_all, axis=1)
        #TODO delete
        from sklearn.metrics import confusion_matrix
        good_states = [i for i in range(n_state) if (Z_all==i).sum()>=1800/epoch_time]
        Zp_all = Zp_all[:,good_states]
        Zp_all = Zp_all/Zp_all.sum(axis=1, keepdims=True)
        Z_all = np.argmax(Zp_all, axis=1)
        ids = pd.notna(Z_all)&pd.notna(sleep_stages_all)
        cf = confusion_matrix(Z_all[ids], sleep_stages_all[ids]-1)
        cf = cf[:,:5]
        sort_val1 = np.argmax(cf,axis=1)
        sort_val2 = cf[range(len(cf)), sort_val1]
        sort_ids = np.lexsort((sort_val2, sort_val1))
        Zp_all = Zp_all[:,sort_ids]
        Z_all = np.argmax(Zp_all, axis=1)
        n_state = Zp_all.shape[1]
        ###
        print(f'n_state = {n_state}')

        for i in tqdm(range(len(df))):
            sid = df.HashID.iloc[i]
            age = int(round(df.Age.iloc[i]))
            sex = df.Sex.iloc[i][0].upper()
            Z = Z_all[sids==sid]
            Zp = Zp_all[sids==sid]
            df_feat = df_feat_all[df_feat_all.HashID==sid]
            with open(f'../step2_dementia_HMM_binary/spectrograms/spec_{sid}.pickle', 'rb') as ff:
                sleep_stages, spec, freq, epoch_start_sec = pickle.load(ff)#, ch_names, start_time
                start_time = datetime.datetime(2000,1,1,hour=22,minute=30)#TODO
                sleep_stages = sleep_stages[epoch_time]
                spec_db = 10*np.log10(spec[epoch_time])
                freq = freq[epoch_time]
                epoch_start_sec = epoch_start_sec[epoch_time]
            assert np.array_equal(sleep_stages, sleep_stages_all[sids==sid], equal_nan=True)

            tt = np.arange(len(sleep_stages))*epoch_time/3600
            xticks = np.arange(int(tt.max())+1)
            xticklabels = [(start_time+datetime.timedelta(hours=int(x))).strftime('%H:%M') for x in xticks]
            assert n_state<=10

            xoffset = -0.06*1.2
            plt.close()
            fig = plt.figure(figsize=(13,7.8))
            gs = GridSpec(3,2, height_ratios=[3,4,4], width_ratios=[5,2.1])

            # AASM sleep stage
            gs_stage = gs[0,0].subgridspec(3,1, height_ratios=[5,n_state,3], hspace=0.15)
            ypad = 0.1
            ax = fig.add_subplot(gs_stage[0,0]); ax0 = ax
            """
            cc = 0
            sleep_stages2 = np.array(sleep_stages)
            sleep_stages2[np.isnan(sleep_stages2)] = -1
            sleep_stages2 = sleep_stages2.astype(int)
            for k,l in groupby(sleep_stages2):
                ll = len(list(l))
                if k>=1:
                    c = aasm_stage_colors[aasm_stages[k-1]]
                    ax.fill_between(
                        tt[[cc,cc+ll-1]], [0,0], [k,k],
                        facecolor=c, edgecolor='none')
                    ax.plot(tt[[cc,cc+ll-1]], [k,k], lw=1.5, color=c)
                cc += ll
            """
            ax.step(tt, sleep_stages, where='post', color='k', lw=1)
            for si, stage in enumerate(aasm_stages):
                sleep_stages2 = np.array(sleep_stages).astype(float)
                sleep_stages2[sleep_stages2!=si+1] = np.nan
                ax.step(tt, sleep_stages2, where='post', color=aasm_stage_colors[stage], lw=2)
            ax.set_yticks([1,2,3,4,5])
            ax.set_yticklabels(['N3', 'N2', 'N1', 'R', 'W'])
            ax.set_ylim(1-ypad, 5+ypad)
            ax.yaxis.grid(True)
            ax.text(xoffset, 1, 'a', ha='right', va='center', transform=ax.transAxes, fontweight='bold')
            ax.text(xoffset, 0.8, f'{age}{sex}', ha='left', va='top', transform=ax.transAxes)
            plt.setp(ax.get_xticklabels(), visible=False)
            sns.despine()

            # new sleep stage
            ax = fig.add_subplot(gs_stage[1,0], sharex=ax0)
            """
            cc = 0
            for z,l in groupby(Z):
                ll = len(list(l))
                c = new_state_colors(z)
                ax.fill_between(
                    tt[[cc,cc+ll-1]], [0,0], [z+1,z+1],
                    facecolor=c, edgecolor='none')
                ax.plot(tt[[cc,cc+ll-1]], [z+1,z+1], lw=1.5, color=c)
                cc += ll
            """
            ax.step(tt, Z+1, where='post', color='k', lw=1)
            for z in Z:
                Z2 = np.array(Z).astype(float)
                Z2[Z!=z] = np.nan
                ax.step(tt, Z2+1, where='post', color=new_state_colors(z), lw=2)
            ax.set_yticks(np.arange(n_state)+1)
            ax.set_ylim(1-ypad, n_state+ypad)
            ax.set_ylabel('New States')
            ax.yaxis.grid(True)
            ax.text(xoffset, 1, 'b', ha='right', va='top', transform=ax.transAxes, fontweight='bold')
            plt.setp(ax.get_xticklabels(), visible=False)
            sns.despine()

            # new sleep stage, hypnodensity
            ax = fig.add_subplot(gs_stage[2,0], sharex=ax0)
            Zp_cumsum = np.c_[np.zeros(len(tt)), np.cumsum(Zp,axis=1)]
            for z in range(n_state):
                ax.fill_between(tt, Zp_cumsum[:,z], Zp_cumsum[:,z+1], facecolor=new_state_colors(z), edgecolor='none')#, alpha=)
            ax.set_yticks([])
            ax.text(xoffset, 1, 'c', ha='right', va='top', transform=ax.transAxes, fontweight='bold')
            plt.setp(ax.get_xticklabels(), visible=False)
            sns.despine()

            # EEG spectrogram
            gs_spec = gs[1,0].subgridspec(3,1, hspace=0.04)
            for chi in range(len(chi2chid)):
                ax = fig.add_subplot(gs_spec[chi,0], sharex=ax0)
                ax.imshow(spec_db[:,chi2chid[chi]].mean(axis=1).T,
                    aspect='auto', origin='lower',
                    cmap='turbo', vmin=-5, vmax=15,
                    extent=(tt.min(), tt.max(), freq.min(), freq.max()))
                ax.set_ylabel(ch_names_combined[chi], rotation=0, ha='right')
                if chi==0:
                    ax.set_yticks([1,5,10,13,20])
                else:
                    ax.set_yticks([1,5,10,13])
                if chi==0:
                    ax.text(xoffset, 1, 'd', ha='right', va='top', transform=ax.transAxes, fontweight='bold')
                plt.setp(ax.get_xticklabels(), visible=False)
                sns.despine()

            # feature heatmap
            ax = fig.add_subplot(gs[2,0], sharex=ax0)
            ax.imshow(df_feat[feat_cols].values.T,
                aspect='auto', origin='upper',
                cmap='Greys', vmin=0, vmax=1,
                extent=(tt.min(), tt.max(), 0, len(feat_cols)),
                interpolation='none')
            for y in np.arange(len(feat_cols)+1):
                ax.axhline(y, color=(204/255,204/255,204/255), lw=0.7)
            ax.set_yticks(np.arange(len(feat_cols))+0.5, labels=feat_cols2[::-1])
            ax.text(xoffset, 1, 'e', ha='right', va='bottom', transform=ax.transAxes, fontweight='bold')
            ax.set_xticks(xticks, labels=xticklabels)
            sns.despine()

            gs_psd = gs[:,1].subgridspec(2,len(chi2chid))
            for chi, chids in enumerate(chi2chid):
                # PSD by AASM stages
                if chi==0:
                    ax = fig.add_subplot(gs_psd[0,chi])
                    ax0 = ax
                else:
                    ax = fig.add_subplot(gs_psd[0,chi], sharex=ax0, sharey=ax0)
                for si, stage in enumerate(aasm_stages):
                    c = aasm_stage_colors[stage]
                    stage_ids = sleep_stages==si+1
                    if np.sum(stage_ids)>=2:
                        ax.plot(freq, spec_db[stage_ids][:,chids].mean(axis=(0,1)), c=c, lw=1.5, alpha=0.7)
                ax.text(0.5,0.9,ch_names_combined[chi],ha='center',va='top',transform=ax.transAxes, fontweight='bold')
                ax.set_xticks([1,5,10,13,20], labels=['1','5','10','13','20Hz'])
                ax.xaxis.grid(True)
                #ax.set_xlabel('Hz')
                if chi==0:
                    #ax.set_ylabel('dB')
                    ax.text(xoffset*2/1.2, 1, 'f', ha='right', va='center', transform=ax.transAxes, fontweight='bold')
                else:
                    plt.setp(ax.get_ylabel(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                plt.setp(ax.get_xticklabels(), visible=False)
                sns.despine()

                # PSD by new states
                ax = fig.add_subplot(gs_psd[1,chi], sharex=ax0, sharey=ax0)
                for z in range(n_state):
                    c = new_state_colors(z)
                    stage_ids = Z==z
                    if np.sum(stage_ids)>=2:
                        ax.plot(freq, spec_db[stage_ids][:,chids].mean(axis=(0,1)), c=c, lw=1.5, alpha=0.7)
                ax.text(0.5,0.9,ch_names_combined[chi],ha='center',va='top',transform=ax.transAxes, fontweight='bold')
                ax.set_xticks([1,5,10,13,20], labels=['1','5','10','13','20Hz'])
                ax.xaxis.grid(True)
                #ax.set_xlabel('Hz')
                if chi==0:
                    #ax.set_ylabel('dB')
                    ax.text(xoffset*2/1.2, 1, 'g', ha='right', va='center', transform=ax.transAxes, fontweight='bold')
                else:
                    plt.setp(ax.get_ylabel(), visible=False)
                    plt.setp(ax.get_yticklabels(), visible=False)
                sns.despine()

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.06, hspace=0.08)
            #plt.show()
            plt.savefig(os.path.join(figure_folder, f'sleep_signal_{epoch_time}s_{sid}.png'), bbox_inches='tight', pad_inches=0.01, dpi=300)
        import pdb;pdb.set_trace()
    
    
if __name__=='__main__':
    main()

