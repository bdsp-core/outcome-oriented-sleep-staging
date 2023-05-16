from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, argrelmax, argrelmin
from scipy.stats import linregress, mode, pearsonr
import mne
import yasa
import logging
from scipy.fftpack import next_fast_len
from scipy import signal
from scipy.interpolate import interp1d
from mne.filter import filter_data
from sklearn.ensemble import IsolationForest
from yasa.numba import _detrend, _rms
from yasa.io import set_log_level
from yasa.detection import _check_data_hypno
from yasa.spectral import stft_power
from yasa.others import trimbothstd, moving_transform, _merge_close


def get_spindle_peak_freq(spec, freq, sleep_stages_epochs, freq_range=[11,15]):
    N2_ids = np.where(sleep_stages_epochs==2)[0]
    if len(N2_ids)<5:
        #TODO based on age norm
        raise ValueError
        
    ids = (freq>=10)&(freq<=20)
    freq = freq[ids]
    spec = spec[...,ids]
    spec_db = 10*np.log10(spec)
    spec_db_N2 = spec_db[N2_ids].mean(axis=0)

    peak_freqs = []
    for chi in range(spec_db_N2.shape[0]):
        slope, intercept, r, p, se = linregress(freq, spec_db_N2[chi])
        spec_ = spec_db_N2[chi]-freq*slope-intercept
        spec_ = savgol_filter(spec_, 45, 3, mode='nearest')
        peak_id = argrelmax(spec_)[0]
        peak_id = peak_id[(freq[peak_id]<=freq_range[1])&(freq[peak_id]>=freq_range[0])]
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


def my_spindle_detect(signals, sleep_stages, Fs, ch_names, include=None, freq_sp=[11,16], thresh={'corr':0.65, 'rel_pow':0.2, 'rms':1.5}, verbose=False, rel_pow_all=None, mcorr_all=None, mrms_all=None, return_precomputed=False, compute_sp_char=True):
    #amp = thresh.pop('amp')
    
    #res = yasa.spindles_detect( signal, sf=Fs, ch_names=ch_names,
    #                hypno=sleep_stages, include=include, freq_sp=freq_sp, freq_broad=[1,30],
    #                duration=[0.5,2], min_distance=500,
    #                thresh=thresh, multi_only=False, remove_outliers=False,
    #                verbose='INFO' if verbose else 'ERROR')

    ################################
    data=signals
    sf=Fs
    hypno=sleep_stages
    freq_broad=[1,30]
    duration=[0.5,2]
    min_distance=500
    multi_only=False
    remove_outliers=False
    verbose = 'INFO' if verbose else 'ERROR'
    logger = logging.getLogger("yasa")
    
    set_log_level(verbose)

    # Check detection thresholds
    if "rel_pow" not in thresh.keys():
        thresh["rel_pow"] = 0.20
    if "corr" not in thresh.keys():
        thresh["corr"] = 0.65
    if "rms" not in thresh.keys():
        thresh["rms"] = 1.5
    do_rel_pow = thresh["rel_pow"] not in [None, "none", "None"]
    do_corr = thresh["corr"] not in [None, "none", "None"]
    do_rms = thresh["rms"] not in [None, "none", "None"]
    n_thresh = sum([do_rel_pow, do_corr, do_rms])
    assert n_thresh >= 1, "At least one threshold must be defined."
    
    (data, sf, ch_names, hypno, include, mask, n_chan, n_samples, bad_chan) = _check_data_hypno(
        data, sf, ch_names, hypno, include
    )
    
    if type(freq_sp[0]) in [float, int]:
        freq_sp = [freq_sp]*n_chan

    # If all channels are bad
    if sum(bad_chan) == n_chan:
        logger.warning("All channels have bad amplitude. Returning None.")
        return None
        
    df = pd.DataFrame()
    if rel_pow_all is None or mcorr_all is None or mrms_all is None:
        # Filtering
        nfast = next_fast_len(n_samples)

        # Initialize empty output dataframe

        rel_pow_all = []
        mcorr_all = []
        mrms_all = []
        for i in range(n_chan):
            # First, skip channels with bad data amplitude
            if bad_chan[i]:
                continue
                
            # 1) Broadband bandpass filter (optional -- careful of lower freq for PAC)
            data_broad = filter_data(data[i], sf, freq_broad[0], freq_broad[1], method="fir", verbose='ERROR')
            # 2) Sigma bandpass filter
            # The width of the transition band is set to 1.5 Hz on each side,
            # meaning that for freq_sp = (12, 15 Hz), the -6 dB points are located at
            # 11.25 and 15.75 Hz.
            data_sigma = filter_data(
                data[i], sf, freq_sp[i][0], freq_sp[i][1],
                l_trans_bandwidth=1.5, h_trans_bandwidth=1.5,
                method="fir", verbose='ERROR'
            )

            # Hilbert power (to define the instantaneous frequency / power)
            analytic = signal.hilbert(data_sigma, N=nfast)[:n_samples]
            inst_phase = np.angle(analytic)
            inst_pow = np.square(np.abs(analytic))
            inst_freq = sf / (2 * np.pi) * np.diff(inst_phase, axis=-1)

            # Compute the pointwise relative power using interpolated STFT
            # Here we use a step of 200 ms to speed up the computation.
            # Note that even if the threshold is None we still need to calculate it
            # for the individual spindles parameter (RelPow).
            f, t, Sxx = stft_power(
                data_broad, sf, window=2, step=0.2, band=freq_broad, interp=False, norm=True
            )
            idx_sigma = np.logical_and(f >= freq_sp[i][0], f <= freq_sp[i][1])
            rel_pow = Sxx[idx_sigma].sum(0)

            # Let's interpolate `rel_pow` to get one value per sample
            # Note that we could also have use the `interp=True` in the
            # `stft_power` function, however 2D interpolation is much slower than
            # 1D interpolation.
            func = interp1d(t, rel_pow, kind="cubic", bounds_error=False, fill_value=0)
            t = np.arange(n_samples) / sf
            rel_pow = func(t)

            if do_corr:
                _, mcorr = moving_transform(
                    x=data_sigma, y=data_broad,
                    sf=sf, window=0.3, step=0.1,
                    method="corr", interp=True,
                )
            if do_rms:
                _, mrms = moving_transform(
                    x=data_sigma, sf=sf, window=0.3, step=0.1, method="rms", interp=True
                )
            rel_pow_all.append(rel_pow)
            mcorr_all.append(mcorr)
            mrms_all.append(mrms)

    result_cols = ['Start', 'End', 'Duration', 'Stage']
    if compute_sp_char:
        result_cols = ['Start', 'Peak', 'End', 'Duration', 'Amplitude', 'RMS',
        'AbsPower', 'RelPower', 'Frequency', 'Oscillations', 'Symmetry',
        'Stage']
    result_cols.extend(['Channel', 'IdxChannel'])
    
    for i in range(n_chan):
        rel_pow = rel_pow_all[i]
        mcorr = mcorr_all[i]
        mrms = mrms_all[i]
        
        # Boolean vector of supra-threshold indices
        idx_sum = np.zeros(n_samples)
        if do_rel_pow:
            idx_rel_pow = (rel_pow >= thresh["rel_pow"]).astype(int)
            idx_sum += idx_rel_pow
            logger.info("N supra-theshold relative power = %i", idx_rel_pow.sum())
        if do_corr:
            idx_mcorr = (mcorr >= thresh["corr"]).astype(int)
            idx_sum += idx_mcorr
            logger.info("N supra-theshold moving corr = %i", idx_mcorr.sum())
        if do_rms:
            # Let's define the thresholds
            if hypno is None:
                thresh_rms = mrms.mean() + thresh["rms"] * trimbothstd(mrms, cut=0.10)
            else:
                thresh_rms = mrms[mask].mean() + thresh["rms"] * trimbothstd(mrms[mask], cut=0.10)
            # Avoid too high threshold caused by Artefacts / Motion during Wake
            thresh_rms = min(thresh_rms, 10)
            logger.info("Moving RMS threshold = %.3f", thresh_rms)
            idx_mrms = (mrms >= thresh_rms).astype(int)
            idx_sum += idx_mrms
            logger.info("N supra-theshold moving RMS = %i", idx_mrms.sum())

        # Make sure that we do not detect spindles outside mask
        if hypno is not None:
            idx_sum[~mask] = 0

        # The detection using the three thresholds tends to underestimate the
        # real duration of the spindle. To overcome this, we compute a soft
        # threshold by smoothing the idx_sum vector with a ~100 ms window.
        # Sampling frequency = 100 Hz --> w = 10 samples
        # Sampling frequecy = 256 Hz --> w = 25 samples = 97 ms
        w = int(0.1 * sf)
        # Critical bugfix March 2022, see https://github.com/raphaelvallat/yasa/pull/55
        idx_sum = np.convolve(idx_sum, np.ones(w), mode="same") / w
        # And we then find indices that are strictly greater than 2, i.e. we
        # find the 'true' beginning and 'true' end of the events by finding
        # where at least two out of the three treshold were crossed.
        where_sp = np.where(idx_sum > (n_thresh - 1))[0]

        # If no events are found, skip to next channel
        if not len(where_sp):
            logger.warning("No spindle were found in channel %s.", ch_names[i])
            continue

        # Merge events that are too close
        if min_distance is not None and min_distance > 0:
            where_sp = _merge_close(where_sp, min_distance, sf)

        # Extract start, end, and duration of each spindle
        sp = np.split(where_sp, np.where(np.diff(where_sp) != 1)[0] + 1)
        idx_start_end = np.array([[k[0], k[-1]] for k in sp]) / sf
        sp_start, sp_end = idx_start_end.T
        sp_dur = sp_end - sp_start

        # Find events with bad duration
        good_dur = np.logical_and(sp_dur > duration[0], sp_dur < duration[1])

        # If no events of good duration are found, skip to next channel
        if all(~good_dur):
            logger.warning("No spindle were found in channel %s.", ch_names[i])
            continue

        # Initialize empty variables
        sp_sta = np.zeros(len(sp))
            
        for j in np.arange(len(sp))[good_dur]:
            # Important: detrend the signal to a
            # Sleep stage
            if hypno is not None:
                sp_sta[j] = hypno[sp[j]][0]
        sp_params = {
            "Start": sp_start,
            "End": sp_end,
            "Duration": sp_dur,
            "Stage": sp_sta,
        }
            
        if compute_sp_char:
            sp_amp = np.zeros(len(sp))
            sp_freq = np.zeros(len(sp))
            sp_rms = np.zeros(len(sp))
            sp_osc = np.zeros(len(sp))
            sp_sym = np.zeros(len(sp))
            sp_abs = np.zeros(len(sp))
            sp_rel = np.zeros(len(sp))
            sp_pro = np.zeros(len(sp))
            # sp_cou = np.zeros(len(sp))

            # Number of oscillations (number of peaks separated by at least 60 ms)
            # --> 60 ms because 1000 ms / 16 Hz = 62.5 m, in other words, at 16 Hz,
            # peaks are separated by 62.5 ms. At 11 Hz peaks are separated by 90 ms
            distance = 60 * sf / 1000

            for j in np.arange(len(sp))[good_dur]:
                # Important: detrend the signal to avoid wrong PTP amplitude
                sp_x = np.arange(data_broad[sp[j]].size, dtype=np.float64)
                sp_det = _detrend(sp_x, data_broad[sp[j]])
                sp_amp[j] = np.ptp(sp_det)  # Peak-to-peak amplitude
                sp_rms[j] = _rms(sp_det)  # Root mean square
                sp_rel[j] = np.median(rel_pow[sp[j]])  # Median relative power

                # Hilbert-based instantaneous properties
                sp_inst_freq = inst_freq[sp[j]]
                sp_inst_pow = inst_pow[sp[j]]
                sp_abs[j] = np.median(np.log10(sp_inst_pow[sp_inst_pow > 0]))
                sp_freq[j] = np.median(sp_inst_freq[sp_inst_freq > 0])

                # Number of oscillations
                peaks, peaks_params = signal.find_peaks(
                    sp_det, distance=distance, prominence=(None, None)
                )
                sp_osc[j] = len(peaks)

                # For frequency and amplitude, we can also optionally use these
                # faster alternatives. If we use them, we do not need to compute
                # the Hilbert transform of the filtered signal.
                # sp_freq[j] = sf / np.mean(np.diff(peaks))
                # sp_amp[j] = peaks_params['prominences'].max()

                # Peak location & symmetry index
                # pk is expressed in sample since the beginning of the spindle
                pk = peaks[peaks_params["prominences"].argmax()]
                sp_pro[j] = sp_start[j] + pk / sf
                sp_sym[j] = pk / sp_det.size

            # Create a dataframe
            sp_params.update({
                "Peak": sp_pro,
                "Amplitude":sp_amp,
                "RMS": sp_rms,
                "AbsPower": sp_abs,
                "RelPower": sp_rel,
                "Frequency": sp_freq,
                "Oscillations": sp_osc,
                "Symmetry": sp_sym,
                # 'SOPhase': sp_cou,
            })

        df_chan = pd.DataFrame(sp_params)[good_dur]

        # We need at least 50 detected spindles to apply the Isolation Forest.
        if remove_outliers and df_chan.shape[0] >= 50:
            col_keep = [
                "Duration",
                "Amplitude",
                "RMS",
                "AbsPower",
                "RelPower",
                "Frequency",
                "Oscillations",
                "Symmetry",
            ]
            ilf = IsolationForest(
                contamination="auto", max_samples="auto", verbose=0, random_state=42
            )
            good = ilf.fit_predict(df_chan[col_keep])
            good[good == -1] = 0
            logger.info(
                "%i outliers were removed in channel %s." % ((good == 0).sum(), ch_names[i])
            )
            # Remove outliers from DataFrame
            df_chan = df_chan[good.astype(bool)]
            logger.info("%i spindles were found in channel %s." % (df_chan.shape[0], ch_names[i]))

        df_chan["Channel"] = ch_names[i]
        df_chan["IdxChannel"] = i
        df = pd.concat([df, df_chan], axis=0, ignore_index=True)
    
    # If no spindles were detected, return None
    if df.empty:
        logger.warning("No spindles were found in data. Returning None.")
        res = pd.DataFrame(columns=result_cols)

    else:
        df = df[result_cols]
        # Remove useless columns
        to_drop = []
        if hypno is None:
            to_drop.append("Stage")
        else:
            df["Stage"] = df["Stage"].astype(int)
        # if not coupling:
        #     to_drop.append('SOPhase')
        if len(to_drop):
            df = df.drop(columns=to_drop)

        # Find spindles that are present on at least two channels
        if multi_only and df["Channel"].nunique() > 1:
            # We round to the nearest second
            idx_good = np.logical_or(
                df["Start"].round(0).duplicated(keep=False), df["End"].round(0).duplicated(keep=False)
            ).to_list()
            df = df[idx_good].reset_index(drop=True)

        #res = yasa.SpindlesResults(
        #    events=df, data=data, sf=sf, ch_names=ch_names, hypno=hypno, data_filt=data_sigma
        #)
        ################################

        res = df#res.summary()
        if 'Amplitude' in res.columns:
            res = res[res.Amplitude<150].reset_index(drop=True)
        res = res.sort_values('Start', ignore_index=True, ascending=True)
        
    if return_precomputed:
        return res, rel_pow_all, mcorr_all, mrms_all
    else:
        return res


def my_sw_detect(eeg, sleep_stages, Fs, ch_names, include=None, freq_sw=[0.5, 2], dur_neg=[0.3,1.5], dur_pos=[0.1,1], amp_neg=[40,200], amp_pos=[10,150], amp_ptp=[70, 350], verbose=False):
    sleep_stages2 = np.array(sleep_stages)
    sleep_stages2[np.isnan(sleep_stages2)] = -1
    sleep_stages2 = sleep_stages2.astype(int)
    
    eeg_f = mne.filter.filter_data(eeg, Fs, freq_sw[0], freq_sw[1], verbose=False)
            
    # compute instantaneous variance explained by SWA
    subepoch_size = int(round(2*Fs))+1
    move_var   = pd.DataFrame(eeg.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T
    move_var_f = pd.DataFrame(eeg_f.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T
    move_var_explained = move_var_f/move_var
    swa_thres = 0.8

    cols = ['Start', 'NegPeak', 'MidCrossing', 'PosPeak', 'End', 'Duration',
        'ValNegPeak', 'ValPosPeak', 'PTP', 'SlopeNeg', 'SlopePos',
        'Frequency', 'Stage', 'Channel', 'IdxChannel']
    
    df_res = defaultdict(list)
    for chi in range(len(eeg_f)):
        # find zero-crossing
        ids_zc_down = np.where((eeg_f[chi,:-1]>0)&(eeg_f[chi,1:]<0))[0]
        ids_zc_up   = np.where((eeg_f[chi,:-1]<0)&(eeg_f[chi,1:]>0))[0]
        ids_zc = np.sort(np.unique(np.r_[0, ids_zc_down, ids_zc_up, eeg.shape[-1]-1]))
        for i in range(len(ids_zc)-2):
            start = ids_zc[i]+1
            mid = ids_zc[i+1]+1
            end = ids_zc[i+2]+1
            #if not (start<mid<end):
            #    continue
            #stage_ = mode(sleep_stages2[start:end]).mode[0]
            stage_ = sleep_stages2[mid]
            if include is not None and stage_ not in include:
                continue
            ptp_ = np.ptp(eeg_f[chi,start:end])
            if ptp_<amp_ptp[0] or ptp_>amp_ptp[1]:
                continue
            ve = move_var_explained[chi, start:end].max()
            if ve>1 or ve<swa_thres:
                continue
            
            mid_is_rising = ids_zc[i] in ids_zc_down
            if mid_is_rising:
                neg_wave = eeg_f[chi, start:mid]
                pos_wave = eeg_f[chi, mid:end]
            else:
                neg_wave = eeg_f[chi, mid:end]
                pos_wave = eeg_f[chi, start:mid]
            dur_neg_ = len(neg_wave)/Fs
            dur_pos_ = len(pos_wave)/Fs
            if dur_neg_<dur_neg[0] or dur_neg_>dur_neg[1]:
                continue
            if dur_pos_<dur_pos[0] or dur_pos_>dur_pos[1]:
                continue
            local_max_neg = argrelmax(neg_wave)[0]
            local_min_neg = argrelmin(neg_wave)[0]
            local_max_pos = argrelmax(pos_wave)[0]
            local_min_pos = argrelmin(pos_wave)[0]
            if not (len(local_max_neg)==0 and len(local_min_neg)==1 and len(local_max_pos)==1 and len(local_min_pos)==0):
                continue
            if mid_is_rising:
                neg_peak_idx = start+local_min_neg[0]
                pos_peak_idx = mid+local_max_pos[0]
            else:
                neg_peak_idx = mid+local_min_neg[0]
                pos_peak_idx = start+local_max_pos[0]
            neg_peak_amp = eeg_f[chi,neg_peak_idx]
            pos_peak_amp = eeg_f[chi,pos_peak_idx]
            if -neg_peak_amp<amp_neg[0] or -neg_peak_amp>amp_neg[1]:
                continue
            if pos_peak_amp<amp_pos[0] or pos_peak_amp>amp_pos[1]:
                continue
            ptp_ = pos_peak_amp-neg_peak_amp
            if ptp_<amp_ptp[0] or ptp_>amp_ptp[1]:
                continue
        
            # remove those with high correlation with eye movement
            #corr = np.array([pearsonr(eeg_f[chi,start:end], eog[chi2,start:end])[0] for chi2 in range(len(eog))])
            #corr = np.abs(corr).max()
            #if corr**2>eog_thres:
            #    continue

            if mid_is_rising:
                df_res['SlopeNeg'].append(np.nan)
                df_res['SlopePos'].append(ptp_/(pos_peak_idx-neg_peak_idx)*Fs)
            else:
                df_res['SlopeNeg'].append(ptp_/(pos_peak_idx-neg_peak_idx)*Fs)
                df_res['SlopePos'].append(np.nan)
                
            df_res['Start'].append(start/Fs)
            df_res['NegPeak'].append(neg_peak_idx/Fs)
            df_res['MidCrossing'].append(mid/Fs)
            df_res['PosPeak'].append(pos_peak_idx/Fs)
            df_res['End'].append(end/Fs)
            df_res['Duration'].append((end-start)/Fs)
            df_res['ValNegPeak'].append(neg_peak_amp)
            df_res['ValPosPeak'].append(pos_peak_amp)
            df_res['PTP'].append(ptp_)
            df_res['Frequency'].append(1/df_res['Duration'][-1])
            df_res['Stage'].append(stage_)
            df_res['Channel'].append(ch_names[chi])
            df_res['IdxChannel'].append(chi)


    if len(df_res)==0:
        return pd.DataFrame(columns=cols)
    else:
        df_res = pd.DataFrame(df_res)
        df_res = df_res[cols]
        df_res = df_res.sort_values('Start', ignore_index=True, ascending=True)
        return df_res


def my_rem_detect(loc, roc, sleep_stages, Fs, ch_name, include=None, amplitude=[50,325], duration=[0.3,1.2], freq_rem=[0.5,5], var_thres=0.7, verbose=False):
    cols = ['Start', 'Peak', 'End', 'Duration',
        'LOCAbsValPeak', 'ROCAbsValPeak', 'LOCAbsRiseSlope', 'ROCAbsRiseSlope',
        'LOCAbsFallSlope', 'ROCAbsFallSlope', 'LOCVarExplained', 'ROCVarExplained',
        'Stage', 'Channel', 'IdxChannel']
    verbose = 'INFO' if verbose else 'ERROR'
    logger = logging.getLogger("yasa")
    set_log_level(verbose)

    if np.in1d(sleep_stages, include).sum()==0:
        return pd.DataFrame(columns=cols)
    res = yasa.rem_detect(loc, roc, Fs,
            hypno=sleep_stages, include=include,
            amplitude=amplitude, duration=duration,
            freq_rem=freq_rem, remove_outliers=False,
            verbose=verbose)
    if res is None:
        return pd.DataFrame(columns=cols)

    res = res.summary()
    if len(res)==0:
        return pd.DataFrame(columns=cols)

    res['Channel'] = ch_name
    res['IdxChannel'] = 0

    subepoch_size = int(round(2*Fs))+1
    eog = np.array([loc,roc])
    eog_f = mne.filter.filter_data(eog, Fs, freq_rem[0], freq_rem[1], verbose=False)
    move_var   = pd.DataFrame(eog.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T
    move_var_f = pd.DataFrame(eog_f.T).rolling(subepoch_size, center=True, min_periods=1).var().values.T
    move_var_explained = move_var_f/move_var
        
    for i in range(len(res)):
        start = int(round(res.Start.iloc[i]*Fs))
        end   = int(round(res.End.iloc[i]*Fs))
        res.loc[i, 'LOCVarExplained'] = move_var_explained[0][start:end].mean()
        res.loc[i, 'ROCVarExplained'] = move_var_explained[1][start:end].mean()
    res = res[(res.LOCVarExplained>=var_thres)&(res.ROCVarExplained>=var_thres)].reset_index(drop=True)

    res = res[cols]
    res = res.sort_values('Start', ignore_index=True, ascending=True)

    return res
