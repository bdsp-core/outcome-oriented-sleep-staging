import datetime
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow import keras  # NOTE: for using TEST venv you should import keras by tensorflow backend
sys.path.insert(0, '..')
from mydatafunctions import read_dataset_mgh_bids
sys.path.insert(0, 'CAISR_model')
from ProductGraphSleepNet import *
from utils import *
sys.path.insert(0, 'CAISR_feature_extraction')
from signal_preprocessing import preprocess
from gcn_features import graph_feat_extraction


def load_model(model_path):
    learn_rate = 0.0001
    lr_decay = 0.0
    l1, l2 = 0.001, 0.001
    opt = keras.optimizers.Adam(learning_rate=learn_rate, clipnorm=1)#, decay=lr_decay
    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)

    w = 11
    h = 9   # height
    context = 7 # 21 Note that it should be odd number 
    sample_shape = (context, w, h)  
    conf_adj = 'GL'
    GLalpha = 0.0
    num_of_chev_filters = 128 # 32 or 64, 128
    num_of_time_filters = 128   # 32 or 64 , 128 
    time_conv_strides = 1   
    time_conv_kernel = 3
    num_block = 1
    cheb_k = 3   # 3, 5
    cheb_polynomials = None  
    dropout = 0.60   # 0.6, 0.75 or 0.8
    GRU_Cell = 256  # 256, 512  or 1024
    attn_heads = 40   # 20, 64 or 128, 256

    model, Hfunc = build_ProductGraphSleepNet(
        cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, 
        time_conv_kernel, sample_shape, num_block, opt, conf_adj=='GL', GLalpha, regularizer, 
        GRU_Cell, attn_heads, dropout)

    model.load_weights(model_path)
    #ww = np.array([[w for w in model.weights if w.name==f'graph_wise_attention_network/kernel_{xx}:0'][0].numpy() for xx in range(attn_heads)]).astype(float)
    #ww = ww.reshape(-1,ww.shape[-1])
    return model, Hfunc


def plot_figure(sleep_stages, sleep_stages_pred, spec_db, freq, start_time, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn
    seaborn.set_style('ticks')

    nch = spec_db.shape[1]
    tt = np.arange(spec_db.shape[0])*30/3600
    xticks = np.arange(int(np.floor(tt.max()))+1)
    xticklabels = [(start_time+datetime.timedelta(hours=int(x))).strftime('%H:%M') for x in xticks]

    plt.close()
    fig = plt.figure(figsize=(13,8))
    gs = fig.add_gridspec(2+nch, height_ratios=[1]*2+[3]*nch)

    # sleep stage (human)
    ax = fig.add_subplot(gs[0]); ax0 = ax
    ax.step(tt, sleep_stages, where='post', c='k', lw=1)
    ax.text(0.002, 0, 'Human', ha='left', va='bottom', transform=ax.transAxes)
    ax.set_yticks([1,2,3,4,5], labels=['N3','N2','N1','R','W'])
    ax.set_ylim(0.9, 5.1)
    ax.yaxis.grid(True)
    ax.set_xticks(xticks, labels=xticklabels)
    seaborn.despine()
    plt.setp(ax.get_xticklabels(), visible=False)

    # sleep stage (model)
    ax = fig.add_subplot(gs[1], sharex=ax0, sharey=ax0)
    ax.step(tt, sleep_stages_pred, where='post', c='k', lw=1)
    ax.text(0.002, 0, 'Model', ha='left', va='bottom', transform=ax.transAxes)
    ax.set_yticks([1,2,3,4,5], labels=['N3','N2','N1','R','W'])
    ax.set_ylim(0.9, 5.1)
    ax.yaxis.grid(True)
    ax.set_xticks(xticks, labels=xticklabels)
    seaborn.despine()
    plt.setp(ax.get_xticklabels(), visible=False)

    # spectrogram
    ch_names = ['F', 'C', 'O']
    for chi, chn in enumerate(ch_names):
        ax = fig.add_subplot(gs[2+chi], sharex=ax0)
        ax.imshow(spec_db[:,chi].T, aspect='auto', origin='lower',
                vmin=-10, vmax=20, cmap='jet',
                extent=(tt.min(), tt.max(), freq.min(), freq.max()) )
        ax.set_ylabel(f'{chn}, Hz')
        ax.set_xticks(xticks, labels=xticklabels)
        seaborn.despine()
        if chi!=len(ch_names)-1:
            plt.setp(ax.get_xticklabels(), visible=False)

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def main():
    root_dir = r'D:\projects\outcome-oriented-sleep-staging\clustering_approach_graphnet'
    root_dir_data = r'D:\projects\dataset_HSP\bids'
    do_plot = True

    model_path = os.path.join(root_dir, 'CAISR_model', 'weights_base_mgh_six_channels.h5')
    output_dir = os.path.join(root_dir, 'sleep_staging_results')
    output_figure_dir = os.path.join(root_dir, 'sleep_staging_figures')
    model_type = 'graphsleepnet'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_figure_dir, exist_ok=True)

    # load model
    context = 7 #TODO
    pad = context//2
    model, Hfunc = load_model(model_path)
    labels = {1:'N3', 2:'N2', 3:'N1', 4:'R', 5:'W', -1:'?'}
    window = 30 # sec

    # load mastersheet
    df = pd.read_excel('../data/mastersheet_outcome_deid.xlsx')
    df['DOV'] = pd.to_datetime(df.DOVshifted)
    
    for i in tqdm(range(len(df))):
        pid = df.BDSPPatientID.iloc[i]
        dov = df.DOV.iloc[i]
        dov2 = dov.strftime('%m%d%Y')
        output_path = os.path.join(output_dir, f'stage_{pid}_{dov2}.csv')
        output_path2 = os.path.join(output_dir, f'H_{pid}_{dov2}.npz')
        output_path3 = os.path.join(output_figure_dir, f'spec_{pid}_{dov2}.png')
        if os.path.exists(output_path):
            continue

        # load signal
        signals, sleep_stages, params = read_dataset_mgh_bids(root_dir_data, pid, dov)
        Fs = params['Fs']
        start_time = params['start_time']

        # preprocess
        segs, sleep_stages2, spec, freq, Fs = preprocess(signals, Fs, sleep_stages)

        # get features
        psd, de = graph_feat_extraction(segs, signals.columns, Fs, window)
        image = AddContext(de, context)
        image = np.squeeze(np.array(image))

        # get output
        ypp = model.predict(image, verbose=False)
        yp = ypp.argmax(axis=1)+1
        H = Hfunc.predict(image, verbose=False)
        H = H.reshape(image.shape[0], -1)

        # save
        tt = np.arange(pad,len(yp)+pad)*30
        df_res = pd.DataFrame(data={
            'start':tt, 'end':tt+30,
            'human stage':['?' if pd.isna(x) else labels[int(x)] for x in sleep_stages2[pad:-pad]],
            'pred stage':[labels[x] for x in yp],
            'prob_N3':ypp[:,0],
            'prob_N2':ypp[:,1],
            'prob_N1':ypp[:,2],
            'prob_R':ypp[:,3],
            'prob_W':ypp[:,4],
            })
        df_res.to_csv(output_path, index=False)
        np.savez_compressed(output_path2, H=H)
        
        if do_plot:
            # assumes F3,F4,C3,C4,O1,O2 order
            spec = spec[:,:6]
            ids = freq<=20
            freq = freq[ids]
            spec = spec[...,ids]
            spec_db = 10*np.log10(spec)
            spec_db = np.nanmean(np.array([spec_db[:,::2], spec_db[:,1::2]]), axis=0)
            yp = np.r_[np.zeros(pad)+np.nan, yp, np.zeros(pad)+np.nan]
            plot_figure(sleep_stages2, yp, spec_db, freq, start_time, save_path=output_path3)


if __name__ == '__main__':
    main()

