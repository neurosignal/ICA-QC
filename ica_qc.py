#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEGnet ICA Quality Control Tool

Author: Amit Jaiswal, Megin Oy, Espoo, Finland
Email : amit.jaiswal@megin.fi

This script is meant to check the quality of ICA applied through MEGnet and 
       override the ICA application with user's own selections.
USAGE:
    - Check ICA outputs:
        python megnet_qc_plots.py --results_dir <directory>
    - Apply ICA manually:
        python megnet_qc_plots.py --ica_file <xxx_0-ica_applied.fif> \
                                  --data_file <xxx_raw_tsss.fif> \
                                  --apply_filter --block --apply_ica
    - Help:
        python megnet_qc_plots.py --help
"""
import os
import argparse
import mne
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from time import sleep
from mne.utils import verbose
from multiprocessing import cpu_count
from mne.time_frequency import psd_array_multitaper

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

def write_explained_variance(ica, raw, save_path):
    print("Writing explained variance ratio...")
    exp_var = ica.get_explained_variance_ratio(raw)
    sensors = raw.get_channel_types()
    filename = os.path.join(save_path, "Explained_variance_ratio.csv")
    with open(filename, 'w') as fid:
        if 'grad' not in sensors:
            fid.write("data_file, \tmag\n")
            fid.write(f"{os.path.basename(raw.filenames[0])}, \t{exp_var.get('mag', 'NA')}")
        elif 'mag' not in sensors:
            fid.write("data_file, \tgrad\n")
            fid.write(f"{os.path.basename(raw.filenames[0])}, \t{exp_var.get('grad', 'NA')}")
        else:
            fid.write("data_file, \tgrad, \tmag\n")
            fid.write(f"{os.path.basename(raw.filenames[0])}, \t{exp_var.get('grad', 'NA')}, \t{exp_var.get('mag', 'NA')}")

def create_report(report_file, title):
    from mne import Report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    return Report(title=title, verbose=None)

# def generate_plots(raw, ica, qc_dir, report_ctx, figwidth, block):
#     print("Generating ICA plots...")

#     # Plot ICA components
#     plot_ica_components(raw, ica, qc_dir, report_ctx, figwidth)

#     # Plot ICA source projections
#     plot_ica_sources(raw, ica, qc_dir, report_ctx, figwidth)

#     # Plot EOG/ECG scores
#     plot_scores(raw, ica, qc_dir, report_ctx)

#     # Overlay raw and cleaned
#     plot_overlay(raw, ica, qc_dir, report_ctx, figwidth)

#     # PSDs
#     plot_psd(raw, qc_dir, report_ctx, figwidth)

#     if block:
#         plt.show()
#     else:
#         plt.close('all')

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

def plot_ica_components(raw, ica, qc_dir, report_ctx, figwidth):
    fig = ica.plot_components(title=f'Removed ICs indices: {ica.exclude}', show=False,
                                    picks=range(ica.n_components_), ncols=6)
    fig.set_figwidth(figwidth)
    if report_ctx:
        report_ctx['ifig'] += 1
        report_ctx['object'].add_figure(
            fig, title='ICA components field-maps', section='All components', 
            caption=f"Fig. {report_ctx['ifig']}. Field maps for all ICA components", 
            **report_ctx['add_cfg'])
        report_ctx['object'].save(fname=report_ctx['file'], **report_ctx['save_cfg'])
    else:
        plt.show(block=False);   plt.pause(0.001)
        fig.savefig(f'{qc_dir}/all_component_topomaps.png', dpi='figure', format='png')

def plot_ica_psds(raw, ica, qc_dir, report_ctx, figwidth):
    ica_ts = get_ica_ts_raw(ica, raw)
    ica_psds, ica_freqs = psd_array_multitaper(
                            mne.make_fixed_length_epochs(ica_ts, duration=25).get_data(), 
                            ica_ts.info['sfreq'], bandwidth=2, n_jobs=cpu_count(),  
                            verbose=None)
    if len(ica_psds.shape) > 2: ica_psds = ica_psds.mean(0)
    ica_idx   = [i for i, item in enumerate(ica_ts.get_channel_types()) if item=='misc']
    eocmg_idx = [i for i, item in enumerate(ica_ts.get_channel_types()) if item!='misc']
    fig, ax   = plt.subplots(2,1, sharex=True, figsize=(figwidth, 6))
    for ii, idxs in enumerate([ica_idx, eocmg_idx]):
        ax[ii].plot(ica_freqs, ica_psds[idxs,:].T, 
                    label=[ica_ts.ch_names[i] for i in idxs])
        ax[ii].set_ylabel('Power')
        ax[ii].legend(ncols=int(ica.n_components_//5))
        ax[ii].grid()
    ax[ii].set_xlabel('Frequency')
    fig.tight_layout()  
    if report_ctx:
        report_ctx['ifig'] += 1
        report_ctx['object'].add_figure(
            fig, title='Components spectra', section='All components', 
            caption=f"Fig. {report_ctx['ifig']}. Power spectra of ICA components.", 
            **report_ctx['add_cfg'])
        report_ctx['object'].save(fname=report_ctx['file'], **report_ctx['save_cfg'])
    else:
        fig.savefig(f'{qc_dir}/all_component_psds.png', dpi='figure', format='png')
        plt.close()

def plot_scores(raw, ica, qc_dir, report_ctx):
    fig = ica.plot_scores(show=False)
    fig_path = os.path.join(qc_dir, "ICA_scores.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved: {fig_path}")

    if report_ctx:
        report_ctx['object'].add_figs_to_section(fig=fig, captions="ICA Scores", **report_ctx['add_cfg'])
        
def plot_ica_sources(raw, ica, qc_dir, report_ctx, figwidth, block=True):
    ica_ts = get_ica_ts_raw(ica, raw)
    # fig    = ica.plot_sources(raw, picks=range(ica.n_components_), show=False, 
    #                           block=False, start=0, stop=10)
    fig = ica_ts.plot(duration=10.0, n_channels=len(ica_ts.ch_names), order=None, 
                show=True, block=block, verbose=None,
                color=dict(eog='b', ecg='m', emg='c', 
                           ref_meg='steelblue', misc='k'), 
                scalings = dict(eog=100e-6, ecg=4e-4, 
                                emg=1e-3, ref_meg=1e-12, misc=1e+1) )
    fig.set_figwidth(figwidth)
    if report_ctx:
        report_ctx['ifig'] += 1
        report_ctx['object'].add_figure(
            fig, title='ICA components time-courses', section='All components', 
            caption=f"Fig. {report_ctx['ifig']}. Time courses for all ICA components", 
            **report_ctx['add_cfg'])
        report_ctx['object'].save(fname=report_ctx['file'], **report_ctx['save_cfg'])
    else:
        plt.show(block=False);   plt.pause(0.001)
        fig.savefig(f'{qc_dir}/all_component_time_series.png', dpi='figure', format='png')
    return ica_ts

def plot_ica_properties(raw, ica, qc_dir, report_ctx, figwidth, block=True):
    figs = ica.plot_properties(raw.copy().pick(['meg', 'eeg', 'eog', 'ecg']),
                               picks=range(ica.n_components_), show=False)
    for ii, fig in enumerate(figs):
        if report_ctx:
            report_ctx['ifig'] += 1
            fig.set_figwidth(figwidth)
            report_ctx['object'].add_figure(
                fig, title=f'IC{str(ii).zfill(3)}', 
                section='Component-wise properties', 
                caption=f"Fig. {report_ctx['ifig'] }. "
                "ICA component IC{str(ii).zfill(3)} properties.",
                **report_ctx['add_cfg'])
            report_ctx['object'].save(fname=report_ctx['file'], **report_ctx['save_cfg'])
        else:
            fig.savefig(f'{qc_dir}/IC{str(ii).zfill(3)}_properties.png', dpi='figure', format='png')
            plt.close()

def plot_overlay(raw, ica, qc_dir, report_ctx, figwidth):
    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned, verbose='error')

    fig = raw.plot(duration=10.0, n_channels=30, show=False)
    fig_cleaned = raw_cleaned.plot(duration=10.0, n_channels=30, show=False)

    fig_path = os.path.join(qc_dir, "Raw_overlay.png")
    fig_cleaned_path = os.path.join(qc_dir, "Raw_cleaned_overlay.png")
    fig.savefig(fig_path, dpi=150)
    fig_cleaned.savefig(fig_cleaned_path, dpi=150)
    print(f"Saved: {fig_path}, {fig_cleaned_path}")

    if report_ctx:
        report_ctx['object'].add_figs_to_section(fig=fig, captions="Original Raw", **report_ctx['add_cfg'])
        report_ctx['object'].add_figs_to_section(fig=fig_cleaned, captions="ICA Cleaned Raw", **report_ctx['add_cfg'])

@verbose
def plot_all(results_dir=None, ica_file=None, data_file=None, report_file=None,
             apply_filter=False, lfreq=None, hfreq=None, block=False, apply_ica=False,
             figwidth=15, verbose=None):

    if results_dir or not all([ica_file, data_file]):
        print("Searching for ICA and data files...")
        data_file, ica_file = find_files(results_dir)

    raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, preload=True)
    ica = mne.preprocessing.read_ica(ica_file)

    if not apply_filter and raw.info['lowpass'] > 100:
        mne.utils.warn("Lowpass is high; consider applying a filter with --apply_filter.")
        sleep(2)

    if apply_filter:
        lfreq = lfreq or raw.info['highpass']
        hfreq = hfreq or raw.info['lowpass']
        raw = apply_filters(raw, lfreq, hfreq)

    qc_dir = os.path.join(os.path.dirname(ica_file), "MEGnetQCplots", os.path.basename(data_file)[:-4])
    os.makedirs(qc_dir, exist_ok=True)
    print(f"Saving report/plots to: {qc_dir}")

    write_explained_variance(ica, raw, qc_dir)

    if report_file is not False:
        report_file = report_file or os.path.join(qc_dir, "MEGNET_QC_report.html")
        report = create_report(report_file, f"MEGNET QC Report: {os.path.basename(ica_file)}")
        report_context = {
            'file': report_file,
            'object': report,
            'ifig': 0,
            'save_cfg': dict(open_browser=False, overwrite=True, sort_content=False, verbose=None),
            'add_cfg': dict(tags=('ica',), image_format='png', replace=True)
        }
    else:
        report_context = None

    # Now modularize plotting calls
    # generate_plots(raw, ica, qc_dir, report_context, figwidth, block)
    plot_ica_components(raw, ica, qc_dir, report_context, figwidth) 
    plot_ica_psds(raw, ica, qc_dir, report_context, figwidth) 
    plot_ica_sources(raw, ica, qc_dir, report_context, figwidth, block=block)

    if apply_ica:
        print("Applying ICA manually...")
        raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, preload=True)
        ica.apply(raw)
        output_path = data_file.replace(os.path.dirname(data_file), os.path.dirname(ica_file)).replace('.fif', '_ICAmanual.fif')
        raw.save(output_path, overwrite=True, verbose=None)
        print(f"ICA applied and saved to: {output_path}")

    if report_context:
        report_context['object'].save(fname=report_context['file'], **report_context['save_cfg'])
        print(f"QC report saved at: {report_context['file']}")

# def generate_plots(raw, ica, qc_dir, report_context, figwidth, block):
#     # This function can be further split (e.g., for each plot type)
#     # For now, keep it monolithic and insert all plot sections here
#     # You already have logic for components, psds, time-courses, scores, etc.
#     pass  # Put your existing plotting blocks here

def parse_args():
    parser = argparse.ArgumentParser(description='Post-ICA QC visualization tool for MEGnet outputs.')
    parser.add_argument('--results_dir',  '-dir', type=str, help='Path to MEGnet results.')
    parser.add_argument('--ica_file',     '-ica', default=None, type=str, help='Path to ICA-applied file.')
    parser.add_argument('--data_file',    '-data', default=None, type=str, help='Raw MEG file.')
    parser.add_argument('--report_file',  '-report', default=None, type=str, help='HTML report path. False to disable.')
    parser.add_argument('--apply_filter', action='store_true', help='Apply bandpass filter before plotting.')
    parser.add_argument('--lfreq',        type=float, help='Low cutoff for bandpass filter.')
    parser.add_argument('--hfreq',        type=float, help='High cutoff for bandpass filter.')
    parser.add_argument('--apply_ica',    action='store_true', help='Apply ICA manually on raw data.')
    parser.add_argument('--block',        action='store_true', help='Block GUI for plots.')
    return parser.parse_args()

if __name__ == '__main__':
    plt.ioff()
    args = parse_args()
    #%%
    args.results_dir = '/home/amit3/pCloudDrive/DATA/Oncology/VUMc/datasel/ica_tmp_di2/1329323_M2B_5minOD_raw_tsss/'
    plot_all(**vars(args))
