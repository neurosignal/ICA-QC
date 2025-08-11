#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICA Quality Control Tool

Author: Amit Jaiswal, Megin Oy, Espoo, Finland
Email : amit.jaiswal@megin.fi

This script is meant to check the quality of ICA and 
       override the ICA application with user's own selections.
USAGE:
    - Check ICA outputs:
        python ica_qc.py --results_dir <directory>
    - Apply ICA manually:
        python ica_qc.py --ica_file <xxx_0-ica_applied.fif> --data_file <xxx_raw_tsss.fif> --apply_filter --block --apply_ica
    - Help:
        python ica_qc.py --help
"""
import os
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from time import sleep
from mne.utils import verbose
from os import cpu_count
from mne.time_frequency import psd_array_multitaper
from copy import deepcopy
from datetime import datetime

plt.rcParams.update({
    "font.size": 12,
    "axes.xmargin": 0
})
mne.viz.set_browser_backend('matplotlib', verbose=None)
print(__doc__)

def find_files(results_dir):
    results_dir = os.path.abspath(results_dir)
    data_files = glob(f'{results_dir}/*_250srate_meg.fif')
    ica_files = glob(f'{results_dir}/*_0-ica_applied.fif')
    if len(data_files) != 1:
        raise FileNotFoundError(f"Expected one data file but found {len(data_files)}: {data_files}")
    if len(ica_files) != 1:
        raise FileNotFoundError(f"Expected one ICA file but found {len(ica_files)}: {ica_files}")
    print(f'Data file: {data_files[0]}')
    print(f'ICA file : {ica_files[0]}')
    return data_files[0], ica_files[0]

def apply_filters(raw, lfreq, hfreq):
    print(f"Applying bandpass filter: {lfreq} — {hfreq} Hz (for visualization only)")
    picks = ['meg', 'eeg', 'eog', 'ecg']
    filter_params = dict(
        picks=picks, filter_length='auto',
        l_trans_bandwidth=1, h_trans_bandwidth=2.0, n_jobs=cpu_count(),
        method='iir', iir_params=dict(order=2, ftype='butter'),
        phase='zero', fir_window='hamming', fir_design='firwin',
        pad='reflect_limited', skip_by_annotation=('edge', 'bad_acq_skip')
    )
    line_freqs = np.arange(raw.info['line_freq'], hfreq, raw.info['line_freq'])
    if line_freqs.any():
        raw.notch_filter(line_freqs, picks=picks, 
                         notch_widths=2, trans_bandwidth=1.0, n_jobs=cpu_count())
    raw.filter(l_freq=lfreq, h_freq=hfreq, **filter_params)
    return raw

def write_explained_variance(ica, raw, report_ctx):
    print("Writing explained variance ratio...")
    exp_var = ica.get_explained_variance_ratio(raw)
    code_str = ''
    for key in ['grad', 'mag', 'eeg']:
        if key in list(exp_var.keys()):
            code_str += f'{key.upper().ljust(4)} : {exp_var[key]:.2f}\n'
    report_ctx['object'].add_html(f'<pre>{code_str}</pre>', 
                    title='Explained variance ratio', 
                    tags=('ica',), replace=True)  
    del code_str
    report_ctx['object'].save(fname=report_ctx['file'], **report_ctx['save_cfg'])

def create_report(report_file, title):
    from mne import Report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    return Report(title=title, verbose=None)

def get_ica_ts_raw(ica, raw):
    ica_ts    = ica.get_sources(raw, add_channels=None, start=None, stop=None)
    raw_eocmg = raw.copy().pick(['eog', 'ecg', 'emg'])
    infoplus  = mne.create_info( 
                (ica_ts.ch_names + raw_eocmg.ch_names), ica_ts.info['sfreq'], 
                ch_types = (['misc'] * len(ica_ts.ch_names) + \
                            raw_eocmg.get_channel_types()), verbose=None)
    data   = np.concatenate((ica_ts._data, raw_eocmg._data), axis=0)
    ica_ts = mne.io.RawArray(data, infoplus)
    ica_ts.info['bads'] = [ica_ts.ch_names[ii] for ii in ica.exclude]
    return ica_ts

def plot_ica_components(ica, report_ctx, figwidth=15):
    picks = np.unique(ica.get_channel_types()).tolist()
    for pick in picks:
        fig = ica.plot_components(title=f'Removed ICs indices: {ica.exclude}', show=False,
                                        picks=range(ica.n_components_), ncols=6, cnorm=None,
                                        ch_type=pick, image_args=None, psd_args=None, verbose=None)
        fig.set_figwidth(figwidth)
        report_ctx['ifig'] += 1
        report_ctx['object'].add_figure(
            fig, title=f'ICA components field-maps (for {pick.upper()}s)', 
            # section='All components', 
            caption=f"Fig. {report_ctx['ifig']}. Field maps for all ICA components using ({pick.upper()}s)", 
            **report_ctx['add_cfg'])
        report_ctx['object'].save(fname=report_ctx['file'], **report_ctx['save_cfg'])

def plot_ica_psds(raw, ica, report_ctx, figwidth=15):
    ica_ts = get_ica_ts_raw(ica, raw)
    ica_psds, ica_freqs = psd_array_multitaper(
                            mne.make_fixed_length_epochs(ica_ts, duration=25).get_data(), 
                            ica_ts.info['sfreq'], bandwidth=2, n_jobs=cpu_count(),  
                            verbose=None)
    if len(ica_psds.shape) > 2: ica_psds = ica_psds.mean(0)
    ica_idx   = [i for i, item in enumerate(ica_ts.get_channel_types()) if item=='misc']
    eocmg_idx = [i for i, item in enumerate(ica_ts.get_channel_types()) if item!='misc']
    fig, ax   = plt.subplots(2,1, sharex=True, figsize=(figwidth, 10))
    for ii, idxs in enumerate([ica_idx, eocmg_idx]):
        ax[ii].plot(ica_freqs, ica_psds[idxs,:].T, 
                    label=[ica_ts.ch_names[i] for i in idxs])
        ax[ii].set_ylabel('Power')
        ax[ii].legend(ncols=int(ica.n_components_//5))
        ax[ii].grid()
    ax[ii].set_xlabel('Frequency')
    fig.tight_layout()  
    report_ctx['ifig'] += 1
    report_ctx['object'].add_figure(
        fig, title='Components spectra', 
        # section='All components', 
        caption=f"Fig. {report_ctx['ifig']}. Power spectra of ICA components.", 
        **report_ctx['add_cfg'])
    report_ctx['object'].save(fname=report_ctx['file'], **report_ctx['save_cfg'])

def plot_ica_scores(raw, ica, qc_dir, report_ctx):
    fig = ica.plot_scores(show=False)
    fig_path = os.path.join(qc_dir, "ICA_scores.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")

    if report_ctx:
        report_ctx['object'].add_figs_to_section(fig=fig, captions="ICA Scores", **report_ctx['add_cfg'])
        
def plot_ica_properties(raw, ica, report_ctx, figwidth, block=True):
    figs = ica.plot_properties(raw.copy().pick(['meg', 'eeg', 'eog', 'ecg']),
                               picks=range(ica.n_components_), show=False)
    for ii, fig in enumerate(figs):
        report_ctx['ifig'] += 1
        fig.set_figwidth(figwidth)
        report_ctx['object'].add_figure(
            fig, title=f'IC{str(ii).zfill(3)}', 
            section='Component-wise properties', 
            caption=f"Fig. {report_ctx['ifig'] }. "
            "ICA component IC{str(ii).zfill(3)} properties.",
            **report_ctx['add_cfg'])
        report_ctx['object'].save(fname=report_ctx['file'], **report_ctx['save_cfg'])

def plot_overlay(raw, ica, qc_dir, report_ctx, figwidth):
    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned, verbose='error')
    figs = {}
    figs['Original data'] = raw.plot(duration=10.0, n_channels=30, show=False)
    figs['Clean data'] = raw_cleaned.plot(duration=10.0, n_channels=30, show=False)
    for chs in ['grad', 'mag', 'eeg']:
        if chs in np.unique(raw.get_channel_types()).tolist():
            figs[f'Overlap {chs.upper()}'] = ica.plot_overlay(raw, picks=chs, 
                                                              show=False)
    if report_ctx:
        for key in list(figs.keys()):
            fig = figs[key]
            fig.set_figwidth(figwidth)
            report_ctx['object'].add_figure(fig, title=key, 
                section='Overlap plots (before vs. after)', 
                caption=f"Plot for {key}.", **report_ctx['add_cfg'])

def plot_artifact_scores(raw, ica, report_ctx, figwidth):
    for artif in ['ecg', 'eog', 'muscle']:
        try:
            idx, scores = eval(f'ica.find_bads_{artif}(raw)')
            fig = ica.plot_scores(scores, n_cols=None, show=False, 
                                  title=f'{artif.upper()} artifact scores ({idx})')
            fig.set_figwidth(figwidth)
            fig.set_figheight(8 if len(fig.axes)>2 else 5)
            scores = [scores] if isinstance(scores, np.ndarray) else scores
            for ii in range(len(scores)):
                ax = fig.axes[ii]
                bars = ax.patches
                for i in idx:
                    bars[i].set_linewidth(4)
            if report_ctx:
                report_ctx['object'].add_figure(
                    fig, title=f'{artif.upper()} artifact scores ({idx})', 
                    caption=f'{artif.upper()} artifact scores ({idx})', 
                    section='Artifact scores', **report_ctx['add_cfg'])
        except ValueError:
            None
    report_ctx['object'].save(fname=report_ctx['file'], **report_ctx['save_cfg'])

def plot_ica_sources(raw, ica, report_ctx, figwidth=15, block=True):
    ica = deepcopy(ica)
    if block: plt.close('all')
    ica_ts = get_ica_ts_raw(ica, raw)
    exc_before =  [ica._ica_names[i] for i in ica.exclude]
    if block:
        save_cfg_tmp = report_ctx['save_cfg']
        save_cfg_tmp['open_browser'] = True
        report_ctx['object'].save(fname=report_ctx['file'], **save_cfg_tmp)
        sleep(1)
        print('Reject/de-reject components...')
    fig = ica_ts.plot(duration=10.0, n_channels=len(ica_ts.ch_names), order=None, 
                show=True, block=block, verbose=None,
                color=dict(eog='b', ecg='m', emg='c', 
                           ref_meg='steelblue', misc='k'), 
                scalings = dict(eog=100e-6, ecg=4e-4, 
                                emg=1e-3, ref_meg=1e-12, misc=1e+1) )
    if block:
        if not (ica_ts.info['bads'] == [ica._ica_names[i] for i in ica.exclude]):
            ica.exclude = [ica._ica_names.index(ch) for ch in ica_ts.info['bads']]
            date_str    = datetime.now().strftime("-%Y%m%d-%H%M%S")
            if report_ctx['new_ica_file'] is None:
                new_ica_file = report_ctx['ica_file'].replace('.fif', date_str + '.fif')
            else:
                new_ica_file = report_ctx['new_ica_file']
            ica.save(new_ica_file)
            exc_after =  [ica._ica_names[i] for i in ica.exclude]
            strA = f' | \t\nNew ica solution saved: {new_ica_file}'
    else:
        exc_after, strA = exc_before, ''
    fig.set_figwidth(figwidth)
    report_ctx['ifig'] += 1
    strB = f' | \tExclued before: {exc_before} \tExclued after: {exc_after}'
    strC = f"Fig. {report_ctx['ifig']}. Time courses for all ICA components"
    report_ctx['object'].add_figure(
        fig, title='ICA components time-courses', 
        # section='All components', 
        caption=strC + strB + strA, 
        **report_ctx['add_cfg'])
    report_ctx['object'].save(fname=report_ctx['file'], **report_ctx['save_cfg'])
    return ica_ts

@verbose
def plot_ica_qc(results_dir=None, ica_file=None, data_file=None, new_ica_file=None,
                apply_filter=False, lfreq=None, hfreq=None, block=False, 
                apply_ica=False, figwidth=15, verbose=None):

    if results_dir and not ica_file and not data_file:
        print("Searching for ICA and data files...")
        data_file, ica_file = find_files(results_dir)
    elif results_dir and ica_file and not data_file:
        print("Searching for data file...")
        data_file, _ = find_files(results_dir)
    elif results_dir and not ica_file and data_file:
        print("Searching for ICA solution...")
        _, ica_file = find_files(results_dir)

    raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, preload=True)
    ica = mne.preprocessing.read_ica(ica_file)

    if not apply_filter and raw.info['lowpass'] > 100:
        mne.utils.warn("Lowpass is high; consider applying a filter with --apply_filter.")
        sleep(2)

    if apply_filter:
        lfreq = lfreq or raw.info['highpass']
        hfreq = hfreq or raw.info['lowpass']
        raw = apply_filters(raw, lfreq, hfreq)

    qc_dir = os.path.join(os.path.dirname(ica_file), "ICA_QC", os.path.basename(data_file)[:-4])
    os.makedirs(qc_dir, exist_ok=True)
    print(f"Saving report/plots to: {qc_dir}")

    report_file = os.path.join(qc_dir, "ICA_QC_report.html")
    report = create_report(report_file, f"ICA QC Report: {os.path.basename(ica_file)}")
    report_context = {
        'data_file'   : data_file,
        'ica_file'    : ica_file,
        'new_ica_file': new_ica_file,
        'file'        : report_file,
        'object'  : report,
        'ifig'    : 0,
        'save_cfg': dict(open_browser=False, overwrite=True, sort_content=False, verbose=None),
        'add_cfg' : dict(tags=('ica',), image_format='png', replace=True)
    }
    
    write_explained_variance(ica, raw, report_context)

    plot_ica_components(ica, report_context, figwidth)
    
    plot_ica_psds(raw, ica, report_context, figwidth) 
        
    plot_ica_properties(raw, ica, report_context, figwidth)
    
    plot_artifact_scores(raw, ica, report_context, figwidth)
    
    plot_overlay(raw, ica, qc_dir, report_context, figwidth)
    
    plot_ica_sources(raw, ica, report_context, figwidth, block=block)   
    
    if apply_ica:
        print("Applying ICA manually...")
        raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, preload=True)
        ica.apply(raw)
        output_path = data_file.replace(os.path.dirname(data_file), 
                                        os.path.dirname(ica_file)).replace('.fif', 
                                                                           '_ICAmanual.fif')
        raw.save(output_path, overwrite=True, verbose=None)
        print(f"\nNew ICA-applied data saved to: {output_path}")

    if report_context:
        report_context['save_cfg']['open_browser'] = True
        report_context['object'].save(fname=report_context['file'], **report_context['save_cfg'])
        print(f"QC report saved at: {report_context['file']}")

def parse_args():
    parser = argparse.ArgumentParser(description='Post-ICA QC visualization tool for MEGnet (or other ICA pipelines) outputs.')
    parser.add_argument('--results_dir',  '-dir', type=str, help='Path to MEGnet (or other ICA pipelines) results.')
    parser.add_argument('--ica_file',     '-ica', default=None, type=str, help='Path to ICA-applied file.')
    parser.add_argument('--data_file',    '-data', default=None, type=str, help='Raw MEG file.')
    parser.add_argument('--new_ica_file', '-nica', default=None, type=str, help='ICA file name if to be written.')
    parser.add_argument('--apply_filter', action='store_true', help='Apply bandpass filter before plotting.')
    parser.add_argument('--lfreq',        type=float, help='Low cutoff for bandpass filter.')
    parser.add_argument('--hfreq',        type=float, help='High cutoff for bandpass filter.')
    parser.add_argument('--apply_ica',    action='store_true', help='Apply ICA manually on raw data.')
    parser.add_argument('--block',        action='store_true', help='Block GUI for plots.')
    return parser.parse_args()

if __name__ == '__main__':
    plt.ioff()
    args = parse_args()
    #% %
    # args.results_dir = '/home/amit3/pCloudDrive/DATA/Oncology/VUMc/datasel/ica_tmp_di2/1329323_M2B_5minOD_raw_tsss/'
    # args.block = True
    # args.ica_file = '/home/amit3/pCloudDrive/DATA/Oncology/VUMc/datasel/ica_tmp_di2/1329323_M2B_5minOD_raw_tsss/1329323_M2B_5minOD_raw_tsss_0-ica_applied-20250811-170305.fif'
    plot_ica_qc(**vars(args))
    plt.close('all')
