#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Compute sensor-level statistics for FPVSWORDS, e.g. ttests
for electrode groups, laterality.
==========================================

OH, March 2024
"""

import FPVS_functions as Ff
import sys

from os import path as op

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from scipy.stats import ttest_rel

from copy import deepcopy

from importlib import reload

import mne
from mne.evoked import EvokedArray

import config_fpvswords as config

reload(config)


reload(Ff)


print(mne.__version__)

conds = {
    # "face": ["face"],
    "english": ["cn", "nw", "pw"],
    # "french": ["cn", "nw", "pw"],
    # "slow": ["cn", "nw", "pw"],
    # "extraslow": ["pw"],
    # "extrafast": ["pw"]
}

resps = ["base", "odd"]

# channel ROIs
my_picks = {}
for roi in config.channel_ROIs:
    my_picks[roi] = config.channel_ROIs[roi]

# add channel groups
sbj_path = op.join(  # get one subject
    config.data_path, config.map_subjects[config.do_subjs[0]][0])
fname_evo = op.join(
    sbj_path,
    "AVE",
    "SumTopo%s_%s_%s%s" % ('Base', 'face', 'face', "-ave.fif")
)
evo = mne.read_evokeds(fname_evo, 0)
# get channel names for channel groups
for ch_type in ['eeg', 'grad', 'mag']:
    tmp = deepcopy(evo)
    tmp.pick(picks=ch_type)
    my_picks[ch_type + 'rms'] = tmp.ch_names

groups = Ff.get_MEG_ROI_channel_names(config.meg_selections, tmp.info)
my_picks.update(groups)

plt.ion()


def sensor_stats(sbj_ids_all):
    """Sensor stats across sbj_ids."""

    print(*sbj_ids_all)

    evo_list = {}  # dicts of list of read evoked data
    roi_data = {}  # dicts of list of data for channel groups
    for cond in conds:  # conditions
        evo_list[cond], roi_data[cond] = {}, {}
        for ev_type in conds[cond]:
            evo_list[cond][ev_type], roi_data[cond][ev_type] = {}, {}
            for resp in resps:
                evo_list[cond][ev_type][resp], roi_data[cond][ev_type][resp] = [], {}
                for metric in list(config.channel_ROIs.keys()):
                    roi_data[cond][ev_type][resp][metric] = []

    laterality = {}  # L-R differences
    stat, pv = {}, {}  # stats for laterality
    signif = {}  #  # count significant responses
    for cond in conds:  # conditions
        if 'extra' in cond:
            sbj_ids = Ff.remove_subjects_extras(sbj_ids_all.copy())
        else:
            sbj_ids = sbj_ids_all
        for ev_type in conds[cond]:
            print("Condition: %s, %s." % (cond, ev_type))

            for resp in resps:
                txt = resp[0].upper() + resp[1:]

                for [ss, sbj_id] in enumerate(sbj_ids):  # across all subjects
                    # path to subject's data
                    sbj_path = op.join(
                        config.data_path, config.map_subjects[sbj_id][0])

                    fname_evo = op.join(
                        sbj_path,
                        "AVE",
                        "SumTopo%s_%s_%s%s" % (txt, cond, ev_type, "-ave.fif"),
                    )
                    print(fname_evo)
                    psd_sum = mne.read_evokeds(fname_evo, 0)

                    evo_list[cond][ev_type][resp].append(psd_sum)

                evos = deepcopy(evo_list[cond][ev_type][resp])

                roi_data[cond][ev_type][resp] = get_amps_channel_types(
                    evos, my_picks)

                columns = [str(i) for i in sbj_ids]
                n_sbj = len(columns)
                for roi in my_picks:
                    data = roi_data[cond][ev_type][resp][roi]
                    data = np.array(data)[:, np.newaxis].T

                    dframe = pd.DataFrame(data=data, columns=columns)

                    fig = plt.figure()
                    sns.barplot(dframe, color="black")
                    fname = "bar_indiv_%s_%s_%s_%s.pdf" % (
                        cond, ev_type, resp, roi)
                    fig_fname = op.join(
                        config.grandmean_path, "Figures", fname)
                    plt.savefig(fig_fname, transparent=True)
                    plt.close()

                # grouped bar plots for individual z-scores
                plot_groups = {'OT': ['OT_L', 'OT_R'],
                               'Grad temporal': ['Grad Left-temporal', 'Grad Right-temporal']}
                for group in plot_groups:
                    data = {}
                    n_sbj = len(columns)
                    groups = plot_groups[group]
                    data['sbj'] = columns + columns
                    data['sbj_nr'] = list(np.arange(n_sbj)) + list(np.arange(n_sbj))
                    data['roi'] = n_sbj * [groups[0]] + n_sbj * [groups[1]]
                    data['Z'] = np.array(roi_data[cond][ev_type][resp][groups[0]] + roi_data[cond][ev_type][resp][groups[1]])
                    dframe = pd.DataFrame(data=data)

                    fig = plt.figure()
                    sns.barplot(dframe, x='sbj_nr', y='Z', hue='roi')
                    fname = "bargrouped_indiv_%s_%s_%s_%s.pdf" % (
                        cond, ev_type, resp, group)
                    fig_fname = op.join(
                        config.grandmean_path, "Figures", fname)
                    print('Saving figure to %s.' % fig_fname)
                    plt.savefig(fig_fname, transparent=True)
                    plt.close()

                    # percentage of lateralised responses
                    data1 = data['Z'][:n_sbj]
                    data2 = data['Z'][n_sbj:2*n_sbj+1]
                    data_lat = data1 - data2  # L-R
                    n_lat = np.where(data_lat>0)[0].size  # number of left-lateralised subjects
                    laterality[cond, ev_type, resp, group] = 100*(n_lat/n_sbj)

                    alternative = 'two-sided'  # what type of t-test, 'less'|'greater'|'two-sided'
                    stat[cond, ev_type, resp, group], pv[cond, ev_type, resp, group] = ttest_rel(data1, data2, alternative=alternative)

                    where1 = np.where(data1 > 1.64)[0]
                    where2 = np.where(data2 > 1.64)[0]
                    signif[cond, ev_type, resp, group, 'lh'] = where1.size
                    signif[cond, ev_type, resp, group, 'rh'] = where2.size
                    signif[cond, ev_type, resp, group, 'lr'] = np.unique(np.concatenate([where1, where2])).size  # left or right

                # Topographies for individual subjects
                evos = evo_list[cond][ev_type][resp]
                topolist = []
                for evo in evos:
                    topo = evo.data[:, evo.time_as_index(0.)].squeeze()
                    # normalise individual topographies for better plotting
                    topo /= np.abs(topo).max()
                    topolist.append(topo)

                evo = deepcopy(evos[0])
                evo.resample(sfreq=1000)
                topos_evo = EvokedArray(
                    np.array(topolist).T, evo.info, tmin=0.0, nave=1)

                scalings = dict(eeg=1., mag=1., grad=1.)
                for ch_type in ['eeg', 'grad', 'mag']:
                    topos_evo.plot_topomap(
                        times=topos_evo.times, ch_type=ch_type, scalings=scalings, time_unit='ms', time_format='%01d')
                    fname = "topos_indiv_%s_%s_%s_%s.pdf" % (
                        cond, ev_type, resp, ch_type)
                    fig_fname = op.join(
                        config.grandmean_path, "Figures", fname)
                    plt.savefig(fig_fname, transparent=True)
                    plt.close()

                # Correlations with behaviour

            plt.close('all')

            # channel_ROIs["OT_R"] = ["EEG055", "EEG060"]  # we don't have P10, PO10, PO12
            # channel_ROIs["OT_L"] = ["EEG047", "EEG056"]  # we don't have P9, PO9, PO11
            # channel_ROIs["MO"] = ["EEG063", "EEG064", "EEG062"]  # we only have O2, IZ, Oz, O1

            # meg_selections = [
            #     "Left-occipital",
            #     "Right-occipital",
            #     "Left-temporal",
            #     "Right-temporal",
            # ]
            # for sel in meg_selections:
            #     channel_ROIs["Mag " + sel] = []
            #     channel_ROIs["Grad " + sel] = []

    for cond in conds:
        for ev_type in conds[cond]:
            for resp in resps:
                for group in plot_groups:
                    print('\n###\n%s, %s, %s, %s: %.1f left-lateralised\n' % (cond, ev_type, resp, group, laterality[cond, ev_type, resp, group]) )

                    print('Stats:')
                    print('%.2f, %g\n' % (stat[cond, ev_type, resp, group], pv[cond, ev_type, resp, group]))

                    print('# signifiant responses:')
                    print('LH | RH | LR: %d | %d | %d' % (signif[cond, ev_type, resp, group, 'lh'], signif[cond, ev_type, resp, group, 'rh'], signif[cond, ev_type, resp, group, 'lr']))

    return


def get_amps_channel_types(evokeds, my_picks):
    """Extract RMS amplitudes across channels per channel type at latency 0.

    Parameters:
    evokeds: list of instances of Evoked
        Evokeds objects to extract amplitudes from, e.g. for peak channels
        per channel type.
    my_picks: dict
        The channel groups to be used as objects for "pick". For example,
        {'eeg': ['EEG001', 'EEG002']} or {'meg': 'grads'}. The keys will be
        used in output 'amps', the entries for the argument "picks" in "pick()".

    Returns:
    amps: dict of list
        Dictionary with keys from ""my_picks" with lists (amplitudes per channel group).
    """

    amps = {}
    for my_pick in my_picks:
        amps[my_pick] = []

    for evoked in evokeds:
        for my_pick in my_picks:
            evo = deepcopy(evoked)

            evo.pick(picks=my_picks[my_pick])

            idx0 = evo.time_as_index(0.0)

            rms = np.sqrt((evo.data[:, idx0] ** 2).mean())

            amps[my_pick].append(rms)

    return amps


# get all input arguments except first
# if number not in config list, do all of them
if (len(sys.argv) == 1) or (
    int(sys.argv[1]) > np.max(list(config.map_subjects.keys()))
):
    # IDs don't start at 0
    sbj_ids_all = config.do_subjs

else:
    # get list of subjects IDs to process
    sbj_ids_all = [int(aa) for aa in sys.argv[1:]]

sensor_stats(sbj_ids_all)
