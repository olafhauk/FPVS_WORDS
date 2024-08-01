#!/imaging/local/software/mne_python/mne1.4.0_1/bin/python
"""
Group stats via cluster-based permutation tests for summed harmonics
==========================================

OH, Mar 2024
"""

### NOT YET STARTED

import sys

from os import path as op

import numpy as np

from scipy.stats import ttest_rel

from matplotlib import pyplot as plt

import pandas as pd

from copy import deepcopy

from importlib import reload

from scipy import stats as stats

import seaborn as sns

import mne
from mne.report import Report
from mne.source_estimate import SourceEstimate
from mne.evoked import EvokedArray
from mne.channels import find_ch_adjacency
from mne.stats import (
    permutation_t_test,
    spatio_temporal_cluster_1samp_test,
    summarize_clusters_stc,
)


# from FPVS_functions import grand_average_evoked_arrays, peak_channels_evoked

import config_fpvswords as config

reload(config)

import FPVS_functions as Ff

reload(Ff)


print(mne.__version__)


figs_dir = "Figures"
suffix = 'pdf'  # suffix for figure filenames

# conditions
# conds = config.do_conds
conds = {
    # "face": ["face"],
    "english": ["cn", "nw", "pw"],
    # "french": ["cn", "nw", "pw"],
    # "slow": ["cn", "nw", "pw"],
    # "extraslow": ["pw"],
    # "extrafast": ["pw"]
}

resps = ["odd", "base"]  # currently only works if both specified
modals = ["evo", "stc"]  # ["evo", "stc"]

# Labels for ROI analysis
subjects_dir = config.subjects_dir

# Labels for ROI analysis
subjects_dir = config.subjects_dir
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir,
                                        verbose=True)

# labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'both',
#                                    subjects_dir=subjects_dir)

# Desikan-Killiany
labels = mne.read_labels_from_annot('fsaverage', 'aparc', 'both',
                                    subjects_dir=subjects_dir)

label_names = {}
# list of list: labels within sub-lists will be combined
# number of items must correpond for 'lh' and 'rh'

# From 'faces', HCPMMP1
# label_names['lh'] = [['L_FFC_ROI-lh'], ['L_VVC_ROI-lh'], ['L_V4_ROI-lh'],
#                      ['L_VMV3_ROI-lh'], ['L_TE2p_ROI-lh'], ['L_V1_ROI-lh']]
# label_names['rh'] = [['R_FFC_ROI-rh'], ['R_VVC_ROI-rh'], ['R_V4_ROI-rh'],
#                      ['R_VMV3_ROI-rh'], ['R_TE2p_ROI-rh'], ['R_V1_ROI-rh']]

# Desikan-Killiany
label_names['lh'] = [['inferiortemporal-lh'], ['middletemporal-lh'],
                      ['superiortemporal-lh']]
label_names['rh'] = [['inferiortemporal-rh'], ['middletemporal-rh'],
                      ['superiortemporal-rh']]

# get subset of labels specified in labels_ATL
my_labels = {'lh': [], 'rh': []}
for hh in ['lh', 'rh']:
    for nn in label_names[hh]:
        tmp = [label for label in labels if label.name == nn[0]][0]
        if len(nn) > 1:
            for n in nn[1:]:
                tmp = tmp + [label for label in labels if label.name == n][0]
        my_labels[hh].append(tmp)

# for some plots of SNRs
unit_scalings = dict(eeg=1.0, mag=1.0, grad=1.0)

subject = "fsaverage"
src_fname = op.join(
    subjects_dir,
    subject,
    "bem",
    subject + "-ico-" + str(config.src_spacing) + "-src.fif",
)
print("Source space from: %s" % src_fname)
src = mne.read_source_spaces(src_fname)


# Read BEHAVIOURAL DATA
dframe_beh_all = pd.read_excel(config.behav_data_fname, '4Pandas')

# choose only good subjects
all_subjs = dframe_beh_all['Subject']


# variables for correlations
# vars_corr = ['IRSTTimeEng', 'BNTTimeEng', 'MillTimeEng', 'IRSTErrEng', 'BNTAccEng', 'MillAccEng', 'LEAPCompEng', 'LEAPSpeakEng', 'LEAPReadEng']
vars_corr = ['VerbFluIndex', 'IRSTIndex', 'BNTIndex', 'MillIndex'] 

def STC_ROI_Analysis(sbj_ids_all):
    """Group stats via cluster-based permutation."""

    print("Using subjects:")
    print(*sbj_ids_all)

    laterality = {}  # L-R differences
    stat, pv = {}, {}  # stats for laterality
    signif = {}  #  # count significant responses
    for cond in conds:
        if 'extra' in cond:
            sbj_ids = Ff.remove_subjects_extras(sbj_ids_all.copy())
        else:
            sbj_ids = sbj_ids_all
        # need to read data each time because of extra condition
        cond_now = {cond: conds[cond]}
        psds, inst_ori = read_subjects_data(sbj_ids, cond_now, resps, modals)
        
        # get relevant behavioural data
        good_subjs = [np.where(x == all_subjs)[0][0] for x in sbj_ids]
        dframe_beh = dframe_beh_all.copy().iloc[good_subjs]

        for ev_type in conds[cond]:
            for resp in resps:

                stcs = psds[cond][ev_type][resp]["stc"]

                amps = {'lh': [], 'rh': []}
                idx0 = np.abs(stcs[0].times).argmin()
                for hh in ['lh', 'rh']:
                    amps[hh] = {}
                    for ll in my_labels[hh]:
                        amps[hh][ll.name] = []
                        for stc in stcs:
                            stc.data = np.abs(stc.data)  # absolute values
                            aa = mne.source_estimate.extract_label_time_course(
                                stcs=stc, labels=ll, src=src, mode='mean', verbose='warning')
                            aa = aa[0, idx0]
                            amps[hh][ll.name].append(aa)
                # t-test, then plot
                data = {}  # data for plotting
                n_sbj = len(stcs)
                data['sbj_nr'] = []  # list(np.arange(n_sbj)) + list(np.arange(n_sbj))
                data['roi'] = []  # ROI names
                data['data'] = []  # values per ROI
                
                # mean across ROIs
                data_mean = {}
                data_mean['data'] = np.zeros(2 * n_sbj)  # values averaged across ROIs, L and R

                print('\n### Condition: %s  -   Response: %s  -  Event: %s\n' % (cond, resp, ev_type))
                for [li, ll] in enumerate(amps['lh']):
                    print('\n# %s\n' % ll)
                    data1 = amps['lh'][my_labels['lh'][li].name]
                    data2 = amps['rh'][my_labels['rh'][li].name]
                    data3 = [(a-b) for (a,b) in zip(data1, data2)]  # laterality per subject
                    data['sbj_nr'] = data['sbj_nr'] + list(np.arange(n_sbj))
                    data['roi'] = data['roi'] + n_sbj * [my_labels['lh'][li].name]
                    data['data'] = data['data'] + data3
                    
                    # append lists then sum values
                    data_mean['data'] = data_mean['data'] + np.array(data1 + data2)

                    # Correlation with behaviour
                    print('Correlations of laterality with behaviour:')
                    for beh_var in vars_corr:
                        corr_stats = stats.pearsonr(data3, dframe_beh[beh_var])
                        print('%s: r = %f   -   p = %f' % (beh_var, corr_stats[0], corr_stats[1]))

                data_mean['data'] /= (li + 1)  # take average
                data_mean['sbj_nr'] = 2 * list(np.arange(n_sbj))
                data_mean['roi'] = n_sbj * ['LH'] + n_sbj * ['RH']

                alternative = 'two-sided'  # what type of t-test, 'less'|'greater'|'two-sided'
                data1 = data_mean['data'][:n_sbj]
                data2 = data_mean['data'][n_sbj:2*n_sbj+1]
                stat[cond, ev_type, resp], pv[cond, ev_type, resp] = ttest_rel(data1, data2, alternative=alternative)

                
                where1 = np.where(data1 > 1.64)[0]
                where2 = np.where(data2 > 1.64)[0]
                signif[cond, ev_type, resp, 'lh'] = where1.size
                signif[cond, ev_type, resp, 'rh'] = where2.size
                signif[cond, ev_type, resp, 'lr'] = np.unique(np.concatenate([where1, where2])).size  # left or right

                # plot laterality for ROI
                dframe = pd.DataFrame(data=data)

                fig1 = plt.figure()
                sns.barplot(dframe, x='sbj_nr', y='data', hue='roi')
                fname = "roilat_bargrouped_indiv_%s_%s_%s.pdf" % (
                    cond, ev_type, resp)
                fig_fname = op.join(
                    config.grandmean_path, "Figures", fname)
                print('Saving figure to %s.' % fig_fname)
                plt.savefig(fig_fname, transparent=True)
                # plt.close(fig1)

                dframe = pd.DataFrame(data=data_mean)
                fig2 = plt.figure()
                sns.barplot(dframe, x='sbj_nr', y='data', hue='roi')
                fname = "lat_bargrouped_indiv_%s_%s_%s.pdf" % (
                    cond, ev_type, resp)
                fig_fname = op.join(
                    config.grandmean_path, "Figures", fname)
                print('Saving figure to %s.' % fig_fname)
                plt.savefig(fig_fname, transparent=True)
                # plt.close(fig2)

                # percentage of lateralised responses
                data1 = data_mean['data'][:n_sbj]
                data2 = data_mean['data'][n_sbj:2*n_sbj+1]
                data_lat = data1 - data2  # L-R
                n_lat = np.where(data_lat>0)[0].size  # number of left-lateralised subjects
                laterality[cond, ev_type, resp] = 100*(n_lat/n_sbj)

    print('')
    for cond in conds:
        for ev_type in conds[cond]:
            for resp in resps:
                print('\n###\n%s, %s, %s: %f%% left-lateralised' % (cond, ev_type, resp, laterality[cond, ev_type, resp]) )

                print('Stats:')
                print('%f, %f\n' % (stat[cond, ev_type, resp], pv[cond, ev_type, resp]))

                print('# signifiant responses:')
                print('LH | RH | LR: %d | %d | %d' % (signif[cond, ev_type, resp, 'lh'], signif[cond, ev_type, resp, 'rh'], signif[cond, ev_type, resp, 'lr']))


def read_subjects_data(sbj_ids, conds, resps, modals):
    """Read sensor and source space data for summed harmonics.

    Parameters:
    sbj_ids: list of int
        The IDs of subject data to be read.
    conds: dict
        Conditions ('english' etc.) and event types ('pw', 'cn' etc.).
        For example {'english': ['cn', 'nw', 'cn']).
    resps: list of string
        The frequency response types ('base' or 'odd').
    modals: list of string
        The data modalities ('evo' or 'stc').

    Returns:
    psds: dict of dict of dict of dict of list
        Data objects for subjects in sbj_ids.
        [cond][ev_type]['base'/'odd']['evo'/'stc'][sbj]
    inst: instance of data object
        The first data object (for future use of info etc.).
    """
    psds = {}  # dict for all relevant data objects
    for cond in conds:
        psds[cond] = {}
        for ev_type in conds[cond]:
            psds[cond][ev_type] = {}
            for resp in resps:
                psds[cond][ev_type][resp] = {}
                for modal in modals:
                    psds[cond][ev_type][resp][modal] = []

    inst = []
    for cond in conds:  # conditions
        for ev_type in conds[cond]:
            print("Condition: %s, %s." % (cond, ev_type))

            for [ss, sbj_id] in enumerate(sbj_ids):  # across all subjects
                # path to subject's data
                sbj_path = op.join(config.data_path, config.map_subjects[sbj_id][0])

                print("Reading PSD results from evoked files:")

                # fname_evo = op.join(sbj_path, 'AVE', 'HarmBaseEpos_%s%s%s' %
                #                     (cond, freq, '-ave.fif'))

                if "evo" in modals:
                    # Oddball topography of z-scored summed harmonics at centre
                    # frequency:
                    fname_evo = op.join(
                        sbj_path, "AVE", "HarmOdd_%s_%s%s" % (cond, ev_type, "-ave.fif")
                    )
                    print(fname_evo)
                    psd_sum_odd = mne.read_evokeds(fname_evo, 0)
                    if inst == []:
                        inst = psd_sum_odd

                    # Base topography of z-scored summed harmonics at centre
                    # frequency:
                    fname_evo = op.join(
                        sbj_path,
                        "AVE",
                        "HarmBase_%s_%s%s" % (cond, ev_type, "-ave.fif"),
                    )
                    print(fname_evo)
                    psd_sum_base = mne.read_evokeds(fname_evo, 0)

                    psds[cond][ev_type]["base"]["evo"].append(psd_sum_base)
                    psds[cond][ev_type]["odd"]["evo"].append(psd_sum_odd)

                ### STC
                if "stc" in modals:
                    fname_stc = op.join(
                        sbj_path,
                        "STC",
                        "PSDZSumTopoOdd_%s_%s_mph-lh.stc" % (cond, ev_type),
                    )
                    print(fname_stc)
                    stc_odd = mne.read_source_estimate(fname_stc)

                    if inst == []:
                        inst = stc_odd

                    fname_stc = op.join(
                        sbj_path,
                        "STC",
                        "PSDZSumTopoBase_%s_%s_mph-lh.stc" % (cond, ev_type),
                    )
                    print(fname_stc)
                    stc_base = mne.read_source_estimate(fname_stc)

                    psds[cond][ev_type]["base"]["stc"].append(stc_base)
                    psds[cond][ev_type]["odd"]["stc"].append(stc_odd)

    return psds, inst


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

# requires all subjects to average across
STC_ROI_Analysis(sbj_ids_all)
